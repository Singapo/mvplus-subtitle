"""Persistent, shared DeepSeek Flash budget; unknown charges stay reserved.

Rates checked against the official CNY table on 2026-10-15. Reserve at peak
cache-miss prices even when the caller waits for off-peak. Amounts are integer
micro-yuan, so rounding cannot allow spending beyond the configured limit.
"""
from contextlib import contextmanager
from decimal import Decimal
import json
import os
from pathlib import Path
import tempfile
import time
import uuid


class BudgetExceeded(RuntimeError):
    pass


ACCOUNT_ATTRIBUTION_HALTS = (
    'Account debit exceeds this ledger; reconcile shared usage, delayed billing or rates',
    'Observed account debit reached the authorized budget',
    'Existing ledger has no starting account balance; reconcile before new paid requests',
)


def paid_cli_args(path, limit):
    if not path or not limit:
        raise ValueError('Paid workflow requires an explicit shared --budget-file and --budget-cny')
    budget=TranslationBudget(path,limit)
    return ['--budget-file',str(budget.path),'--budget-cny',str(limit)]


class TranslationBudget:
    def __init__(self, path, limit_cny, *, scope=None, scope_limit_cny=None):
        self.path = Path(path).resolve()
        value = Decimal(str(limit_cny))
        if not value.is_finite() or value <= 0:
            raise ValueError('Budget must be finite and positive')
        self.limit = int(value * 1_000_000)
        if self.limit < 1:
            raise ValueError('Budget is too small')
        self.scope=scope
        if (scope is None)!=(scope_limit_cny is None):raise ValueError('Scope and scope limit are required together')
        self.scope_limit=None
        if scope is not None:
            if not isinstance(scope,str) or not scope.strip():raise ValueError('Invalid budget scope')
            scoped_value=Decimal(str(scope_limit_cny))
            if not scoped_value.is_finite() or scoped_value<=0:raise ValueError('Invalid scope budget')
            self.scope_limit=int(scoped_value*1_000_000)
            if self.scope_limit<1:raise ValueError('Scope budget is too small')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._locked():
            if self.path.exists():
                self._read()
            else:
                self._write(dict(version=1, limit_micro_cny=self.limit, requests={}))
            if self.scope is not None:
                state=self._read();scopes=state.setdefault('scopes',{})
                if self.scope in scopes and scopes[self.scope]['limit_micro_cny']!=self.scope_limit:
                    raise ValueError('Scope budget identity/limit changed; refusing to reset spending')
                if self.scope not in scopes:
                    scopes[self.scope]=dict(limit_micro_cny=self.scope_limit,created_at=time.time())
                    self._write(state)

    @contextmanager
    def _locked(self):
        lock = self.path.with_suffix(self.path.suffix + '.lock')
        deadline = time.monotonic() + 10
        while True:
            try:
                fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                break
            except (FileExistsError,PermissionError):
                # A killed worker leaves the marker behind; it is only held for a
                # few milliseconds around read/write, so age is a safe stale test.
                try:
                    if time.time() - lock.stat().st_mtime > 60:
                        lock.unlink(missing_ok=True)
                        continue
                except OSError:
                    pass
                if time.monotonic() >= deadline:
                    raise RuntimeError(f'Budget locked; inspect before retrying: {lock}')
                time.sleep(.02)
        try:
            yield
        finally:
            os.close(fd)
            lock.unlink()

    def _read(self):
        state = json.loads(self.path.read_text(encoding='utf-8'))
        if state['version'] != 1 or state['limit_micro_cny'] != self.limit:
            raise ValueError('Budget identity/limit changed; refusing to reset spending')
        if state.get('account_mode','exclusive') not in ('shared','exclusive'):
            raise ValueError('Invalid account mode')
        for request in state['requests'].values():
            if type(request['charged_micro_cny']) is not int or request['charged_micro_cny'] < 0:
                raise ValueError('Invalid budget ledger')
        return state

    def _write(self, state):
        with tempfile.NamedTemporaryFile(dir=self.path.parent, suffix='.part', delete=False) as f:
            temporary = Path(f.name)
            f.write(json.dumps(state, ensure_ascii=False, indent=2).encode('utf-8'))
        try:
            temporary.replace(self.path)
        finally:
            temporary.unlink(missing_ok=True)

    def reserve(self, prompt, max_tokens, model):
        if model != 'deepseek-flash':
            raise ValueError('Budget rates only support deepseek-flash')
        if type(max_tokens) is not int or not 1 <= max_tokens <= 32768:
            raise ValueError('max_tokens must be between 1 and 32768')
        # UTF-8 bytes upper-bound byte-level tokens; allowance covers chat framing.
        prompt_bound = len(prompt.encode('utf-8')) + 1024
        amount = prompt_bound * 2 + max_tokens * 8
        with self._locked():
            state = self._read()
            if state.get('halt_reason'):
                raise BudgetExceeded('Paid requests halted: '+state['halt_reason'])
            used = sum(r['charged_micro_cny'] for r in state['requests'].values())
            if used + amount > self.limit:
                raise BudgetExceeded(f'Budget cap reached: reserved/spent {used/1e6:.6f}, '
                                     f'next <= {amount/1e6:.6f}, cap {self.limit/1e6:.2f} CNY')
            if self.scope is not None:
                if state.get('scopes',{}).get(self.scope,{}).get('limit_micro_cny')!=self.scope_limit:
                    raise ValueError('Scope budget changed')
                scoped_used=sum(r['charged_micro_cny'] for r in state['requests'].values() if r.get('scope')==self.scope)
                if scoped_used+amount>self.scope_limit:
                    raise BudgetExceeded(f'Scope budget cap reached: {self.scope}')
            request_id = uuid.uuid4().hex
            state['requests'][request_id] = dict(
                status='reserved_or_unknown', charged_micro_cny=amount,
                prompt_token_bound=prompt_bound, max_tokens=max_tokens,
                created_at=time.time())
            if self.scope is not None:state['requests'][request_id]['scope']=self.scope
            self._write(state)
        return request_id

    def increase_limit(self, limit_cny, reason):
        """Apply an explicit authorization without resetting costs or holds."""
        value = Decimal(str(limit_cny))
        if not value.is_finite() or value <= 0 or int(value*1_000_000) <= self.limit:
            raise ValueError('New budget must be a finite increase')
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError('Explicit authorization reason required')
        new_limit = int(value*1_000_000)
        with self._locked():
            state = self._read()
            if any(r['status'] != 'completed' for r in state['requests'].values()):
                raise BudgetExceeded('Unsettled requests; inspect before increasing budget')
            state.setdefault('limit_history', []).append(dict(
                previous_limit_micro_cny=self.limit, new_limit_micro_cny=new_limit,
                reason=reason, at=time.time()))
            state['limit_micro_cny'] = new_limit
            self._write(state)
            self.limit = new_limit
        return self.summary()

    def halt(self, reason):
        with self._locked():
            state = self._read()
            state['halt_reason'] = str(reason)
            self._write(state)

    def resume(self,reason):
        if not isinstance(reason,str) or not reason.strip():raise ValueError('Explicit resume reason required')
        with self._locked():
            state=self._read()
            state.setdefault('resume_history',[]).append(dict(reason=reason,previous_halt=state.pop('halt_reason',None),at=time.time()))
            self._write(state)

    def confirm_shared_account(self,reason):
        """Record confirmed sharing; release only an account-attribution halt."""
        if not isinstance(reason,str) or not reason.strip():raise ValueError('Explicit confirmation reason required')
        with self._locked():
            state=self._read()
            previous=state.get('halt_reason')
            cleared=previous if previous in ACCOUNT_ATTRIBUTION_HALTS else None
            state['account_mode']='shared'
            state.setdefault('account_mode_history',[]).append(dict(
                mode='shared',reason=reason,cleared_halt=cleared,at=time.time()))
            if cleared:state.pop('halt_reason')
            self._write(state)

    def observe_balance(self,balance_cny,initial_balance=None):
        """Observe availability; only exclusive accounts permit debit attribution."""
        value=Decimal(str(balance_cny))
        if not value.is_finite():raise ValueError('Invalid account balance')
        current=int(value*1_000_000)
        with self._locked():
            state=self._read();account=state.get('account',{})
            shared=state.get('account_mode','exclusive')=='shared'
            if not account:
                if state['requests'] and initial_balance is None and not shared:
                    state['halt_reason']='Existing ledger has no starting account balance; reconcile before new paid requests'
                    self._write(state)
                    raise BudgetExceeded(state['halt_reason'])
                anchor=Decimal(str(initial_balance)) if initial_balance is not None else value
                if not anchor.is_finite():raise ValueError('Invalid initial balance')
                account={'initial_micro_cny':int(anchor*1_000_000)}
            elif initial_balance is not None and int(Decimal(str(initial_balance))*1_000_000)!=account['initial_micro_cny']:
                raise ValueError('Cannot reset an existing account baseline')
            debit=max(0,account.get('debit_micro_cny',0),account['initial_micro_cny']-current)
            upper=sum(r['charged_micro_cny'] for r in state['requests'].values())
            account.update(current_micro_cny=current,debit_micro_cny=debit,observed_at=time.time())
            state['account']=account
            # Two cents cover the two rounded displayed balances, not unknown bills.
            if not shared and not state.get('halt_reason') and debit>upper+20_000:
                state['halt_reason']=ACCOUNT_ATTRIBUTION_HALTS[0]
            if not shared and not state.get('halt_reason') and debit>=self.limit:
                state['halt_reason']=ACCOUNT_ATTRIBUTION_HALTS[1]
            self._write(state)
            reason=state.get('halt_reason')
        if reason:raise BudgetExceeded(reason)

    def settle(self, request_id, usage,receipt=None):
        # Missing/invalid usage must never release the reservation, even on retry.
        prompt = usage.get('prompt_tokens')
        output = usage.get('completion_tokens')
        if any(type(n) is not int or n < 0 for n in (prompt, output)):
            self.halt('Missing/invalid API usage; retain reservations and inspect before resuming')
            raise RuntimeError('Missing API usage; reservation retained; stop paid requests')
        with self._locked():
            state = self._read()
            request = state['requests'][request_id]
            if prompt > request['prompt_token_bound'] or output > request['max_tokens']:
                state['halt_reason'] = 'API token usage exceeded reserved bounds'
                self._write(state)
                raise RuntimeError('API exceeded token bounds; reservation retained; inspect billing')
            # Deliberately conservative: peak, no discount for automatic cache hits.
            request.update(status='completed', usage=usage, charged_micro_cny=prompt * 2 + output * 8)
            if receipt:request.update(receipt)
            self._write(state)

    def summary(self):
        with self._locked():
            state = self._read()
        result=dict(limit_cny=self.limit/1e6,
                    conservative_spent_or_reserved_cny=sum(
                        r['charged_micro_cny'] for r in state['requests'].values())/1e6,
                    requests=len(state['requests']),
                    unknown_requests=sum(r['status'] != 'completed' for r in state['requests'].values()),
                    account_mode=state.get('account_mode','exclusive'),
                    halt_reason=state.get('halt_reason'),
                    observed_account_debit_cny=state.get('account',{}).get('debit_micro_cny',0)/1e6 if state.get('account') else None)
        if self.scope is not None:
            result.update(scope=self.scope,scope_limit_cny=self.scope_limit/1e6,
                scope_spent_or_reserved_cny=sum(r['charged_micro_cny'] for r in state['requests'].values() if r.get('scope')==self.scope)/1e6)
        return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['status','resume','confirm-shared'])
    parser.add_argument('--budget-file',required=True);parser.add_argument('--budget-cny',required=True)
    parser.add_argument('--reason')
    args=parser.parse_args();budget=TranslationBudget(args.budget_file,args.budget_cny)
    if args.action=='resume':budget.resume(args.reason)
    if args.action=='confirm-shared':budget.confirm_shared_account(args.reason)
    print(json.dumps(budget.summary(),ensure_ascii=False))
