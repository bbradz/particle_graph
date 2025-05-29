import torch
from typing import List, Union, Dict, Tuple, Callable, Any

Token = Union[int, str, Tuple[int, int], Dict[str, Any]]
Predicate = Callable[[int], bool]

class TorchGrammar:
    def __init__(self):
        self.raw_rules: Dict[str, List[Token]] = {}
        self.symbol_to_index: Dict[str, int] = {}
        self.index_to_symbol: Dict[int, str] = {}

    def add_object(self, name: str, tokens: List[Token]):
        self.raw_rules[name] = tokens
        if name not in self.symbol_to_index:
            idx = len(self.symbol_to_index)
            self.symbol_to_index[name] = idx
            self.index_to_symbol[idx] = name

    @staticmethod
    def expand_literal(tok: Token) -> List[int]:
        if isinstance(tok, tuple):
            return list(range(tok[0], tok[1] + 1))
        if isinstance(tok, int):
            return [tok]
        raise ValueError(f"Invalid literal token: {tok}")

    @staticmethod
    def wrap_pred(raw: Any) -> Predicate:
        if callable(raw):
            return raw
        return lambda cnt, m=raw: cnt >= m


class TorchGrammarBatchParser:
    def __init__(self, grammar: TorchGrammar, batch_size: int, device="cuda"):
        self.g      = grammar
        self.bs     = batch_size
        self.dev    = torch.device(device)
        self.max_d  = 16
        self.max_l  = 128
        self.max_n  = 32

        # [batch, depth, (obj_idx, ptr, repeat_cnt)]
        self.stack = torch.full(
            (self.bs, self.max_d, 3),
            -1,
            dtype=torch.long,
            device=self.dev
        )
        self.ptr   = torch.zeros(self.bs, dtype=torch.long, device=self.dev)

        self.seqs  = torch.full(
            (self.bs, self.max_l),
            -1,
            dtype=torch.long,
            device=self.dev
        )
        self.lens  = torch.zeros(self.bs, dtype=torch.long, device=self.dev)

        self.next_cache = torch.full(
            (self.bs, self.max_n),
            -1,
            dtype=torch.long,
            device=self.dev
        )
        self.next_lens  = torch.zeros(self.bs, dtype=torch.long, device=self.dev)

    def set_batch(self, names: List[str]):
        for i, nm in enumerate(names):
            idx = self.g.symbol_to_index[nm]
            self.stack[i, 0] = torch.tensor([idx, 0, 0], device=self.dev)
            self.ptr[i]      = 1
        self._refresh_next()

    def _pop_and_advance(self, i: int):
        self.ptr[i] -= 1
        if self.ptr[i].item() == 0:
            return

        (o_idx, o_ptr, o_rep), depth = self.stack[i, self.ptr[i]-1].tolist(), self.ptr[i]-1
        obj_name = self.g.index_to_symbol[o_idx]
        token    = self.g.raw_rules[obj_name][o_ptr]

        if isinstance(token, dict):
            sub, raw_pred = next(iter(token.items()))
            pred = self.g.wrap_pred(raw_pred)
            if not pred(o_rep):
                return  # still allowed to repeat

        # advance past this token/group
        self.stack[i, depth, 1] = o_ptr + 1
        self.stack[i, depth, 2] = 0

    def _refresh_next(self):
        self.next_lens.zero_()
        for i in range(self.bs):
            while self.ptr[i].item() > 0:
                o_idx, o_ptr, o_rep = self.stack[i, self.ptr[i]-1].tolist()
                obj_name = self.g.index_to_symbol[o_idx]
                tokens   = self.g.raw_rules[obj_name]

                if o_ptr >= len(tokens):
                    self._pop_and_advance(i)
                    continue

                tok   = tokens[o_ptr]
                valid = self._compute_valid(tok, o_rep, tokens, o_ptr)
                if valid:
                    k = len(valid)
                    self.next_cache[i, :k] = torch.tensor(valid, device=self.dev)
                    self.next_lens[i]      = k
                    break

                self.ptr[i] -= 1  # nothing valid here → pop

    def _compute_valid(self, tok: Token, rep_cnt: int, tokens: List[Token], idx: int) -> List[int]:
        if isinstance(tok, dict):
            sub, raw_pred = next(iter(tok.items()))
            pred = self.g.wrap_pred(raw_pred)
            valid: List[int] = []
            if not pred(rep_cnt):
                first = self.g.raw_rules[sub][0]
                valid += self.g.expand_literal(first)
            if idx + 1 < len(tokens):
                nxt = tokens[idx + 1]
                if isinstance(nxt, str):
                    first = self.g.raw_rules[nxt][0]
                    valid += self.g.expand_literal(first)
                else:
                    valid += self.g.expand_literal(nxt)
            return valid

        if isinstance(tok, str):
            first = self.g.raw_rules[tok][0]
            return self.g.expand_literal(first)

        return self.g.expand_literal(tok)

    def fast_is_valid(self, batch: torch.Tensor) -> torch.Tensor:
        mask    = torch.zeros(self.bs, dtype=torch.bool, device=self.dev)
        cache   = self.next_cache.clone()
        lengths = self.next_lens.clone()

        for i in range(self.bs):
            tok = batch[i].item()
            nt  = lengths[i].item()
            if nt and (cache[i, :nt] == tok).any():
                mask[i] = True
                pos     = self.lens[i].item()
                if pos < self.max_l:
                    self.seqs[i, pos] = tok
                    self.lens[i]    += 1

                o_idx, o_ptr, o_rep = self.stack[i, self.ptr[i]-1].tolist()
                curr = self.g.raw_rules[self.g.index_to_symbol[o_idx]][o_ptr]
                self._apply_token(i, curr, tok, o_rep)

        self._refresh_next()
        return mask

    def _apply_token(self, i: int, curr: Token, tok: int, rep_cnt: int):
        if isinstance(curr, dict):
            sub, raw_pred = next(iter(curr.items()))
            pred = self.g.wrap_pred(raw_pred)
            first = self.g.raw_rules[sub][0]

            if not pred(rep_cnt) and tok in self.g.expand_literal(first):
                depth = self.ptr[i].item()
                sid   = self.g.symbol_to_index[sub]
                self.stack[i, depth]       = torch.tensor([sid, 1, 0], device=self.dev)
                self.ptr[i]               += 1
                self.stack[i, depth-1, 2] += 1
            else:
                self.stack[i, self.ptr[i]-1, 1] += 1
                self.stack[i, self.ptr[i]-1, 2] = 0

        elif isinstance(curr, str):
            depth = self.ptr[i].item()
            sid   = self.g.symbol_to_index[curr]
            self.stack[i, depth] = torch.tensor([sid, 0, 0], device=self.dev)
            self.ptr[i]         += 1

        else:
            self.stack[i, self.ptr[i]-1, 1] += 1
            self.stack[i, self.ptr[i]-1, 2] = 0

    def get_sequences(self) -> List[List[int]]:
        return [self.seqs[i, :self.lens[i]].tolist() for i in range(self.bs)]


if __name__ == "__main__":
    g = TorchGrammar()
    g.add_object("Z", [16, 17])
    g.add_object("Y", [12, 13, {"Z": lambda cnt: cnt >= 2}, 15])
    g.add_object("X", [1, 2, (3, 5), 6])

    p = TorchGrammarBatchParser(g, batch_size=3, device="cuda")
    p.set_batch(["Y", "X", "Y"])

    tests = [
        [12, 1, 12],
        [13, 2, 13],
        [16, 3, 16],
        [17, 5, 17],
        [16, 6, 15],
        [17, 6, 16],
        [16, 6, 16],
        [15, 6, 16],
    ]
    for toks in tests:
        print(p.fast_is_valid(torch.tensor(toks, device="cuda")))
    print(p.get_sequences())


    # Now you should get:
    # tensor([True, True, True], device='cuda:0')
    # tensor([True, True, True], device='cuda:0')
    # tensor([True, True, True], device='cuda:0')
    # tensor([True, False, True], device='cuda:0')
    # tensor([True, True, True], device='cuda:0')
    # tensor([True, False, False], device='cuda:0')
    # tensor([False, False, False], device='cuda:0')
    # [[12, 13, 16, 17, 16, 17], [1, 2, 3, 6], [12, 13, 16, 17, 15]]