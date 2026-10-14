import math
import torch
import torch.distributed as dist


#######################################################
# Scion: https://github.com/LIONS-EPFL/scion/blob/f58a393e010395a5913176be5673d8839caa6285/scion.py
#######################################################


class Norm(object):
    def lmo(self, g):
        raise NotImplementedError

    def init(self, w):
        raise NotImplementedError

    def norm(self, w):
        raise NotImplementedError


class ColNorm(Norm):
    """
    Column-wise normalization.

    Args:
        normalized (bool, optional): If True, normalizes by the input dimension. Use True only for non-input layers.
        transpose (bool, optional): If True, transposes input before normalization. Use True for embedding layers
                which store weights as (vocab_size, embedding_dim).
    """
    def __init__(self, normalized=False, transpose=False):
        self.normalized = normalized
        self.transpose = transpose

    def lmo(self, g):
        eps = 1e-8
        if self.transpose:
            g = g.transpose(0, 1) 
        rms_values = 1/math.sqrt(g.size(0))*torch.sqrt(torch.sum(g ** 2, dim=0, keepdim=True))
        if self.normalized:
            rms_values *= g.size(1)
        g = g / (rms_values + eps)
        if self.transpose:
            g = g.transpose(0, 1) 
        return g

    def init(self, w):
        dtype = w.data.dtype
        if self.transpose:
            w.data = w.data.transpose(0, 1)
        torch.nn.init.normal_(w.data)
        w.data /= w.norm(dim=0, keepdim=True)
        w.data *= math.sqrt(w.size(0))
        if self.normalized:
            w.data /= w.size(1)
        w.data = w.data.to(dtype=dtype)
        if self.transpose:
            w.data = w.data.transpose(0, 1)
        return w

    def norm(self, w):
        x = w.transpose(0, 1) if self.transpose else w
        if x.ndim != 2:
            raise ValueError(f"ColNorm.norm expects 2D tensor, got {tuple(w.shape)}")
        col_rms = torch.sqrt(torch.mean(x.float() ** 2, dim=0))
        val = col_rms.max()
        if self.normalized:
            val = val * x.size(1)
        return val.to(dtype=w.dtype)


class RowNorm(Norm):
    """
    Row-wise normalization.

    Args:
        normalized (bool, optional): If True, normalizes by the input dimension. Use False only for the input layer.
        transpose (bool, optional): If True, transposes input before normalization. Use True for embedding layers
                which store weights as (vocab_size, embedding_dim).
    """
    def __init__(self, normalized=True, transpose=False):
        self.normalized = normalized
        self.transpose = transpose

    def lmo(self, g):
        eps = 1e-8
        if self.transpose:
            g = g.transpose(0, 1) 
        rms_values = torch.sqrt(torch.sum(g ** 2, dim=-1, keepdim=True))
        if self.normalized:
            rms_values *= math.sqrt(g.size(-1))
        g = g / (rms_values + eps)
        if self.transpose:
            g = g.transpose(0, 1) 
        return g

    def init(self, w):
        dtype = w.data.dtype
        if self.transpose:
            w.data = w.data.transpose(0, 1)
        torch.nn.init.normal_(w.data)
        w.data /= w.norm(dim=-1, keepdim=True)
        if self.normalized:
            w.data /= math.sqrt(w.size(-1))
        w.data = w.data.to(dtype=dtype)
        if self.transpose:
            w.data = w.data.transpose(0, 1)       
        return w

    def norm(self, w):
        if w.ndim in (0, 1):
            return BiasRMS().norm(w)
        x = w.transpose(0, 1) if self.transpose else w
        if x.ndim != 2:
            raise ValueError(f"RowNorm.norm expects 2D tensor, got {tuple(w.shape)}")
        row_l2 = torch.sqrt(torch.sum(x.float() ** 2, dim=-1))
        val = row_l2.max()
        if self.normalized:
            val = val * math.sqrt(x.size(-1))
        return val.to(dtype=w.dtype)


class BiasRMS(Norm):
    def lmo(self, g):
        eps = 1e-8
        rms_values = torch.sqrt(torch.mean(g ** 2, dim=0, keepdim=True))
        g = g / (rms_values + eps)
        return g

    def init(self, g):
        return torch.nn.init.zeros_(g)

    def norm(self, w):
        if w.ndim == 0:
            return w.abs()
        if w.ndim == 1:
            return torch.sqrt(torch.mean(w.float() ** 2)).to(dtype=w.dtype)
        raise ValueError(f"BiasRMS.norm expects 0D/1D tensor, got {tuple(w.shape)}")


# class L2Norm(Norm):
#     """Global L2 normalization for a tensor."""
#     def lmo(self, g):
#         eps = 1e-8
#         return g / (torch.linalg.vector_norm(g) + eps)

#     def init(self, g):
#         return torch.nn.init.zeros_(g)


class SpectralConv(Norm):
    def __init__(self, steps=5):
        self.steps = steps

    def lmo(self, g):
        g = zeropower_via_newtonschulz5(g.reshape(len(g), -1), steps=self.steps).view(g.shape)
        if g.ndim == 3:    # Conv1d
            out_channels, in_channels, k = g.shape
            g *= (out_channels / in_channels)**0.5 / k
        elif g.ndim == 4:   # Conv2d
            out_channels, in_channels, k, _ = g.shape
            g *= (out_channels / in_channels)**0.5 / (k ** 2)
        return g
    
    def init(self, w):
        w_fp = w.data.double()
        k = w.data.size(2)
        for kx in range(k):
            for ky in range(k):
                torch.nn.init.orthogonal_(w_fp[:,:,kx,ky])
        
        if w.ndim == 3:     # Conv1d
            out_channels, in_channels, k = w_fp.shape
            w_fp.mul_((out_channels / in_channels)**0.5 / k)
        elif w.ndim == 4:     # Conv2d
            out_channels, in_channels, k, _ = w_fp.shape
            w_fp.mul_((out_channels / in_channels)**0.5 / (k ** 2))
        w.data = w_fp.to(dtype=w.data.dtype)
        return w

    def norm(self, w):
        if w.ndim not in (3, 4):
            raise ValueError(f"SpectralConv.norm expects 3D/4D tensor, got {tuple(w.shape)}")
        out_ch, in_ch = w.shape[0], w.shape[1]
        w_flat = w.reshape(out_ch, -1)
        if w.ndim == 3:
            k = w.shape[2]
            scale = math.sqrt(out_ch / in_ch) / k
        else:
            k = w.shape[2]
            scale = math.sqrt(out_ch / in_ch) / (k ** 2)
        sigma = torch.linalg.matrix_norm(w_flat.float(), ord=2)
        return (sigma / scale).to(dtype=w.dtype)


class Spectral(Norm):
    def __init__(self, max=False, normalized=True, steps=5):
        self.max = max
        self.steps = steps
        self.normalized = normalized

    def lmo(self, g):
        g = zeropower_via_newtonschulz5(g.reshape(len(g), -1), steps=self.steps).view(g.shape)
        d_out, d_in = g.shape
        
        if self.normalized:
            scale = (d_out / d_in)**0.5
        else:
            scale = d_out**0.5
        if self.max:
            scale = max(1,scale)
        g *= scale

        return g

    def init(self, w):
        # Section B.3 semi-orthogonal draw. Float32 is sufficient for QR-based
        # initialization and avoids the 2x memory/startup cost of a float64
        # temporary, especially for wide MLP matrices.
        w_fp = w.data.float()
        torch.nn.init.orthogonal_(w_fp)
        d_out, d_in = w_fp.shape
        
        if self.normalized:
            scale = (d_out / d_in)**0.5
        else:
            scale = d_out**0.5
        if self.max:
            scale = max(1,scale)
        w_fp.mul_(scale)
    
        w.data = w_fp.to(dtype=w.data.dtype)
        return w

    def norm(self, w):
        if w.ndim != 2:
            raise ValueError(f"Spectral.norm expects 2D tensor, got {tuple(w.shape)}")
        d_out, d_in = w.shape
        if self.normalized:
            scale = (d_out / d_in) ** 0.5
        else:
            scale = d_out ** 0.5
        if self.max:
            scale = max(1.0, scale)
        sigma = torch.linalg.matrix_norm(w.float(), ord=2)
        return (sigma / scale).to(dtype=w.dtype)


class Sign(Norm):
    def __init__(self, zero_init=False, normalized=True):
        self.zero_init = zero_init
        self.normalized = normalized

    def lmo(self, g):
        d_out, d_in = g.shape
        if self.normalized:
            return (1/d_in)*torch.sign(g)    
        else:
            return torch.sign(g)

    def init(self, w):
        if self.zero_init:
            torch.nn.init.zeros_(w)
        else:
            # Generate -1/fan_in or 1/fan_in uniformly at random
            d_out, d_in = w.shape
            w.data = (torch.randint(0, 2, w.shape, dtype=w.dtype, device=w.device) * 2 - 1)
            if self.normalized:
                w.data *= (1/d_in)
        return w

    def norm(self, w):
        if w.ndim < 2:
            max_abs = w.abs().max()
            return max_abs.to(dtype=w.dtype)
        d_in = w.shape[-1]
        max_abs = w.abs().max()
        if self.normalized:
            max_abs = max_abs * d_in
        return max_abs.to(dtype=w.dtype)


class Auto(Norm):
    def lmo(self, g):
        if g.ndim in [3,4]:
            return SpectralConv().lmo(g)
        elif g.ndim == 2:
            return Spectral().lmo(g)
        elif g.ndim in [0,1]:
            return BiasRMS().lmo(g)

    def init(self, w):
        if w.ndim in [3,4]:
            return SpectralConv().init(w)
        elif w.ndim == 2:
            return Spectral().init(w)
        elif w.ndim in [0,1]:
            return BiasRMS().init(w)

    def norm(self, w):
        if w.ndim in [3, 4]:
            return SpectralConv().norm(w)
        elif w.ndim == 2:
            return Spectral().norm(w)
        elif w.ndim in [0, 1]:
            return BiasRMS().norm(w)
        raise ValueError(f"Auto.norm: unsupported tensor rank {w.ndim} for shape {tuple(w.shape)}")


norm_dict = {
    'ColNorm': ColNorm,
    'RowNorm': RowNorm,
    'BiasRMS': BiasRMS,
    'SpectralConv': SpectralConv,
    'Spectral': Spectral,
    'Sign': Sign,
    'Auto': Auto,
}


class Scion(torch.optim.Optimizer):
    """Scion optimizer implementation.

    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional): Learning rate (default: 1e-3)
        momentum (float, optional): One minus the traditional momentum factor. For example,
            a traditional momentum of 0.9 would be specified as momentum=0.1 here (default: 1.0)
        norm (str, optional): Choice of norm for gradient projection ('Auto', 'SpectralConv', 
            'ColNorm', 'RowNorm', 'BiasRMS', 'Spectral', or 'Sign') (default: 'Auto')
        norm_kwargs (dict, optional): Additional arguments for the norm projection (default: None)
        scale (float, optional): Scale factor for updates (default: 1.0)
        unconstrained (bool, optional): Whether to use unconstrained updates (default: False)
    
    Example:
        >>> radius = 50.0
        >>> optim_groups = [{
        ...     'params': model.transformer.h.parameters(),
        ...     'norm': 'Spectral',
        ...     'norm_kwargs': {},
        ...     'scale': radius,
        ... }, {
        ...     'params': model.lm_head.parameters(),
        ...     'norm': 'Sign',
        ...     'norm_kwargs': {},
        ...     'scale': radius*60.0,
        ... }]
        >>> optimizer = Scion(optim_groups, lr=2**-12, momentum=0.1)
    """
    def __init__(self, params, lr=1e-3, momentum=1.0, norm: str='Auto', norm_kwargs: dict=None, scale=1.0, unconstrained=False):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if norm_kwargs is None:
            norm_kwargs = {}
        defaults = dict(lr=lr, momentum=momentum, scale=scale, unconstrained=unconstrained, norm=norm, norm_kwargs=norm_kwargs)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        # Foreach updates and one batched Newton-Schulz call per matrix shape.
        for group in self.param_groups:
            params = [p for p in group['params'] if p.grad is not None]
            if not params:
                continue
            lr = group['lr']
            momentum = group['momentum']
            scale = group['scale']
            unconstrained = group['unconstrained']
            norm_backend = norm_dict[group['norm']](**group['norm_kwargs'])

            directions = [p.grad for p in params]
            if momentum != 1:
                bufs = []
                for p in params:
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(p.grad)
                    bufs.append(state['momentum_buffer'])
                torch._foreach_mul_(bufs, 1 - momentum)
                torch._foreach_add_(bufs, directions, alpha=momentum)
                directions = bufs

            updates = torch._foreach_mul(_lmo_all(norm_backend, directions), scale)
            data = [p.data for p in params]
            if not unconstrained:
                torch._foreach_mul_(data, 1 - lr)
            torch._foreach_add_(data, updates, alpha=-lr)
        return loss

    def init(self):
        for group in self.param_groups:
            norm_backend = norm_dict[group['norm']](**group['norm_kwargs'])
            init_func = norm_backend.init
            scale = group['scale']
            for p in group['params']:
                init_func(p)
                p.data *= scale


class ScionTrace(Scion):
    """Scion with step-level tracing metrics compatible with sciontrace logs."""

    def __init__(self, *args, trace: bool = False, trace_every: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.trace = trace
        # The local-smoothness (L) terms need an SVD of every weight update and
        # an extra LMO per step, which dominates the step time.  trace_every=N
        # computes them only on every N-th step (the previous gradient and
        # weights are still copied every step, so each L sample stays a
        # consecutive-step difference); trace_every=0 disables them and the
        # stored copies entirely.  The gradient norms (the mu inputs) are
        # logged every step either way.
        self.trace_every = max(0, int(trace_every))
        self.n_steps = 0
        self._last_trace_logs = {}

    @torch.no_grad()
    def pop_trace_logs(self):
        logs = self._last_trace_logs
        self._last_trace_logs = {}
        return logs

    @torch.no_grad()
    def pop_noise_logs(self):
        logs = getattr(self, "_last_noise_logs", {})
        self._last_noise_logs = {}
        return logs

    def _reset_noise_state(self):
        self._gn_state = {
            "m": 0,
            "sum_norm2": 0.0,
            "sum_grads": None,
            "ref_samples": [],
            "loss_count": 0,
            "loss_sum": 0.0,
            "loss_sq_sum": 0.0,
        }

    @torch.no_grad()
    def track_stats(self, cur_loss=None, store_dtype: torch.dtype = torch.bfloat16, store_on_cpu: bool = False):
        if not hasattr(self, "_gn_state"):
            self._reset_noise_state()

        state = self._gn_state
        params = [p for group in self.param_groups for p in group["params"]]
        n_params = len(params)
        if state["sum_grads"] is None or len(state["sum_grads"]) != n_params:
            state["sum_grads"] = [None] * n_params
            state["ref_samples"] = []
            state["m"] = 0
            state["sum_norm2"] = 0.0
            state["loss_count"] = 0
            state["loss_sum"] = 0.0
            state["loss_sq_sum"] = 0.0

        keep_reference = state["m"] < 3
        sample = [] if keep_reference else None
        sample_sq = 0.0
        for i, p in enumerate(params):
            grad = p.grad
            if grad is None:
                if keep_reference:
                    sample.append(None)
                continue
            grad_f32 = grad.detach().to(dtype=torch.float32)
            sample_sq += float(torch.sum(grad_f32 * grad_f32).item())

            acc = state["sum_grads"][i]
            if acc is None:
                state["sum_grads"][i] = grad_f32.clone()
            else:
                acc.add_(grad_f32)

            if keep_reference:
                grad_store = grad.detach()
                if store_on_cpu:
                    grad_store = grad_store.to(device="cpu", dtype=store_dtype)
                else:
                    grad_store = grad_store.to(dtype=store_dtype)
                sample.append(grad_store.clone())

        state["sum_norm2"] += sample_sq
        state["m"] += 1
        if keep_reference:
            state["ref_samples"].append(sample)

        if cur_loss is not None:
            try:
                cur_loss_f = float(cur_loss)
                state["loss_count"] += 1
                state["loss_sum"] += cur_loss_f
                state["loss_sq_sum"] += cur_loss_f * cur_loss_f
            except (TypeError, ValueError):
                pass

    @torch.no_grad()
    def report_stats(self, eps: float = 1e-12):
        state = getattr(self, "_gn_state", None)
        if state is None:
            return None
        m = int(state.get("m", 0))
        if m == 0:
            return None
        samples = state["ref_samples"]

        params = [p for group in self.param_groups for p in group["params"]]
        norm_backends = []
        for group in self.param_groups:
            norm_backend = norm_dict[group["norm"]](**group["norm_kwargs"])
            for _ in group["params"]:
                norm_backends.append(norm_backend)

        # E||g||^2 over stochastic samples
        Eg2 = float(state["sum_norm2"]) / float(m)

        # ||E[g]||^2 using the sample mean gradient
        mean_g2 = 0.0
        sum_grads = state["sum_grads"]
        for i, _ in enumerate(params):
            acc = sum_grads[i]
            if acc is None:
                continue
            mean_grad = acc / float(m)
            mean_g2 += float(torch.sum(mean_grad * mean_grad).item())

        reference_samples = [
            samples[0],
            samples[1] if m > 1 else samples[0],
            samples[2] if m > 2 else samples[min(1, m - 1)],
        ]
        delta_star = [0.0, 0.0, 0.0]
        delta_fro_sq = [0.0, 0.0, 0.0]

        for i, _ in enumerate(params):
            acc = sum_grads[i]
            if acc is None:
                continue

            mean_grad = acc / float(m)
            norm_backend = norm_backends[i]
            for sample_idx, sample in enumerate(reference_samples):
                grad = sample[i]
                grad = (
                    torch.zeros_like(mean_grad)
                    if grad is None
                    else grad.to(device=mean_grad.device, dtype=torch.float32)
                )
                # Do not compare a stochastic gradient against a reference
                # mean that contains that same sample.  The leave-one-out mean
                # is independent of this sample and avoids an m-dependent
                # shrinkage of rho when trace_m is small at large batches.
                reference_grad = (
                    (acc - grad) / float(m - 1)
                    if m > 1
                    else mean_grad
                )
                delta = reference_grad - grad
                delta_fro_sq[sample_idx] += float(torch.sum(delta * delta).item())
                lmo_delta = norm_backend.lmo(delta).to(dtype=torch.float32)
                delta_star[sample_idx] += float(torch.sum(delta * lmo_delta).item())

        if dist.is_available() and dist.is_initialized():
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            world_size = dist.get_world_size()
            for scalar_name in ("Eg2", "mean_g2"):
                val = Eg2 if scalar_name == "Eg2" else mean_g2
                t = torch.tensor(float(val), device=device, dtype=torch.float32)
                dist.all_reduce(t, op=dist.ReduceOp.SUM)
                if scalar_name == "Eg2":
                    Eg2 = float(t.item() / world_size)
                else:
                    mean_g2 = float(t.item() / world_size)
            for values in (delta_star, delta_fro_sq):
                for i, value in enumerate(values):
                    t = torch.tensor(float(value), device=device, dtype=torch.float32)
                    dist.all_reduce(t, op=dist.ReduceOp.SUM)
                    values[i] = float(t.item() / world_size)

        delta_fro = [math.sqrt(max(v, 0.0)) for v in delta_fro_sq]
        rho = [
            float(delta_star[i] / (delta_fro[i] + eps))
            for i in range(3)
        ]
        var_biased = max(Eg2 - mean_g2, 0.0)
        var_unbiased = var_biased * (m / (m - 1)) if m > 1 else 0.0
        std = math.sqrt(max(var_unbiased, 0.0))
        mean_norm = math.sqrt(max(mean_g2, 0.0))
        snr = mean_norm / (std + eps)
        stats = {
            "grad/noise_samples": float(m),
            "grad/noise_E_grad_norm2": float(Eg2),
            "grad/noise_mean_grad_norm2": float(mean_g2),
            "grad/noise_sigma2": float(var_unbiased),
            "grad/noise_sigma": float(std),
            "grad/noise_snr": float(snr),
            "rho/delta_sample_0_fro": float(delta_fro[0]),
            "rho/delta_sample_0_star": float(delta_star[0]),
            "rho/noise_rho_sample_0": rho[0],
            "rho/delta_sample_1_fro": float(delta_fro[1]),
            "rho/delta_sample_1_star": float(delta_star[1]),
            "rho/noise_rho_sample_1": rho[1],
            "rho/delta_sample_2_fro": float(delta_fro[2]),
            "rho/delta_sample_2_star": float(delta_star[2]),
            "rho/noise_rho_sample_2": rho[2],
            "rho/averaged_rho_over_samples": float(sum(rho) / len(rho)),
            "rho/rho_over_averaged_norms": float(sum(delta_star) / (sum(delta_fro) + eps)),
            "rho/reference_samples": float(max(m - 1, 0)),
            "rho/noise_step": float(self.n_steps),
        }

        loss_count = int(state.get("loss_count", 0))
        if loss_count == m and loss_count > 0:
            loss_sum = float(state["loss_sum"])
            loss_sq_sum = float(state["loss_sq_sum"])
            loss_mean = loss_sum / float(loss_count)
            if loss_count > 1:
                loss_var = (loss_sq_sum - ((loss_sum * loss_sum) / float(loss_count))) / float(loss_count - 1)
                loss_var = max(loss_var, 0.0)
            else:
                loss_var = 0.0
            stats["grad/noise_loss_mean"] = loss_mean
            stats["grad/noise_loss_var"] = loss_var

        self._reset_noise_state()
        self._last_noise_logs = stats
        return stats

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.n_steps += 1
        trace = bool(self.trace)
        smooth_enabled = self.trace_every > 0
        smooth_due = smooth_enabled and self.n_steps % self.trace_every == 0
        num = den = grad_norm = dual_grad_norm = num_nuc = den_spec = 0.0
        grad_checksum = 0.0

        if trace and smooth_enabled and self.n_steps == 1:
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    st = self.state[p]
                    st['prev_p'] = p.detach().clone()
                    st['prev_grad'] = p.grad.detach().clone()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            scale = group['scale']
            unconstrained = group['unconstrained']
            norm_backend = norm_dict[group['norm']](**group['norm_kwargs'])
            for p in group['params']:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                raw_p = p.detach().clone() if (not trace or smooth_enabled) else None
                raw_g = g.detach().clone()

                if momentum != 1:
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(g)
                    buf = state['momentum_buffer']
                    buf.mul_(1 - momentum).add_(g, alpha=momentum)
                    g = buf

                lmo = norm_backend.lmo(g)
                update = scale * lmo
                if not unconstrained:
                    p.data.mul_(1 - lr)
                p.data.add_(update, alpha=-lr)

                if trace and not smooth_enabled:
                    grad_norm += raw_g.pow(2).sum().item()
                    dual_grad_norm += (raw_g * lmo).sum().item()
                elif trace and 'prev_grad' in state and 'prev_p' in state:
                    if smooth_due:
                        grad_diff = state['prev_grad'].detach() - raw_g
                        num += grad_diff.pow(2).sum().item()
                        num_nuc += (grad_diff.clone().mul_(norm_backend.lmo(grad_diff))).sum().item()
                    grad_norm += raw_g.pow(2).sum().item()
                    grad_checksum += raw_g.float().sum().item()
                    state['prev_grad'].copy_(raw_g)

                    if smooth_due:
                        delta_p = raw_p.detach() - state['prev_p']
                        den += delta_p.pow(2).sum().item()
                        den_spec = max(den_spec, float(norm_backend.norm(delta_p)))
                    state['prev_p'].copy_(raw_p)

                    dual_grad_norm += (raw_g * lmo).sum().item()

        if trace:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            num_t = torch.tensor([num], device=device, dtype=torch.float32)
            den_t = torch.tensor([den], device=device, dtype=torch.float32)
            num_nuc_t = torch.tensor([num_nuc], device=device, dtype=torch.float32)
            den_spec_t = torch.tensor([den_spec], device=device, dtype=torch.float32)
            grad_norm_t = torch.tensor([grad_norm], device=device, dtype=torch.float32)
            dual_grad_norm_t = torch.tensor([dual_grad_norm], device=device, dtype=torch.float32)
            # grad_checksum_t = torch.tensor([grad_checksum], device=device, dtype=torch.float32)
            ddp_trace_checks = {}
            # if dist.is_available() and dist.is_initialized():
            #     world_size = dist.get_world_size()
            #     ddp_trace_checks["stats/ddp_trace_world_size"] = float(world_size)
                # for name, value in (
                #     ("num_sq", num_t),
                #     ("den_sq", den_t),
                #     ("num_nuc", num_nuc_t),
                #     ("den_spec", den_spec_t),
                #     ("grad_norm_sq", grad_norm_t),
                #     ("grad_checksum", grad_checksum_t),
                # ):
                    # gathered = [torch.zeros_like(value) for _ in range(world_size)]
                    # dist.all_gather(gathered, value)
                    # gathered_values = torch.cat(gathered)
                    # max_abs_diff = gathered_values.max() - gathered_values.min()
                    # ddp_trace_checks[
                    #     f"stats/ddp_trace_{name}_max_abs_diff"
                    # ] = float(max_abs_diff.abs())
            num = float(torch.sqrt(num_t))
            den = float(torch.sqrt(den_t))
            num_nuc = float(num_nuc_t)
            den_spec = float(den_spec_t)
            grad_norm = float(grad_norm_t)
            dual_grad_norm = float(dual_grad_norm_t)
            local_smooth_fro = num / (den + 1e-8)
            local_smooth_spec = num_nuc / (den_spec + 1e-8)
            self._last_trace_logs = {
                "stats/grad_norm_fro_power_1": float(grad_norm),
                "stats/grad_norm_nuc_power_1": float(dual_grad_norm),
                "stats/num_fro": float(num),
                "stats/den_fro": float(den),
                "stats/num_nuc": float(num_nuc),
                "stats/den_spec": float(den_spec),
                "stats/local_smooth_fro": float(local_smooth_fro),
                "stats/local_smooth_spec": float(local_smooth_spec),
                "stats/stats_step": float(self.n_steps),
            }
            if not smooth_due:
                for key in ("stats/num_fro", "stats/den_fro", "stats/num_nuc",
                            "stats/den_spec", "stats/local_smooth_fro",
                            "stats/local_smooth_spec"):
                    del self._last_trace_logs[key]
            self._last_trace_logs.update(ddp_trace_checks)
        return loss


class ScionLight(torch.optim.Optimizer):
    """Memory-efficient variant of the Scion optimizer.
    
    This implementation saves memory by storing only the averaged gradient instead of 
    both the gradient and its average. Note that gradients should not be zeroed since
    p.grad is used directly to store the gradient average.
    
    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional): Learning rate (default: 1e-3)
        momentum (float, optional): One minus the traditional momentum factor. For example,
            a traditional momentum of 0.9 would be specified as momentum=0.1 here (default: 1.0)
        norm (str, optional): Choice of norm for gradient projection ('Auto', 'SpectralConv', 
            'ColNorm', 'RowNorm', 'BiasRMS', 'Spectral', or 'Sign') (default: 'Auto')
        norm_kwargs (dict, optional): Additional arguments for the norm projection (default: None)
        scale (float, optional): Scale factor for updates (default: 1.0)
        unconstrained (bool, optional): Whether to use unconstrained updates (default: False)
    
    Example:
        >>> radius = 50.0
        >>> optim_groups = [{
        ...     'params': model.transformer.h.parameters(),
        ...     'norm': 'Spectral',
        ...     'norm_kwargs': {},
        ...     'scale': radius,
        ... }, {
        ...     'params': model.lm_head.parameters(),
        ...     'norm': 'Sign',
        ...     'norm_kwargs': {},
        ...     'scale': radius*60.0,
        ... }]
        >>> optimizer = ScionLight(optim_groups, lr=2**-12, momentum=0.1)
    """
    def __init__(self, params, lr=1e-3, momentum=1.0, norm: str='Auto', norm_kwargs: dict=None, scale=1.0, unconstrained=False):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if norm_kwargs is None:
            norm_kwargs = {}
        defaults = dict(lr=lr, momentum=momentum, scale=scale, unconstrained=unconstrained, norm=norm, norm_kwargs=norm_kwargs)
        super().__init__(params, defaults)
        # Initialize state
        self._store_grads_in_state()
        # Do not pass `self` through syntactic sugar. We need the
        # argument to not be populated.
        self.register_state_dict_pre_hook(
            type(self)._store_grads_in_state,
        )
        self.register_load_state_dict_post_hook(
            type(self)._load_grads_from_state,
        )

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            scale = group['scale']
            unconstrained = group['unconstrained']
            norm_backend = norm_dict[group['norm']](**group['norm_kwargs'])
            for p in group['params']:
                G = p.grad
                if G is None:
                    continue

                update = scale * norm_backend.lmo(G)
                if not unconstrained:
                    p.data.mul_(1-lr)
                p.data.add_(update, alpha=-lr)
                
                if momentum != 1:
                    G.mul_(1-momentum)
        return loss

    def init(self):
        for group in self.param_groups:
            norm_backend = norm_dict[group['norm']](**group['norm_kwargs'])
            init_func = norm_backend.init
            scale = group['scale']
            for p in group['params']:
                init_func(p)
                p.data *= scale

    def __getstate__(self):
        self._store_grads_in_state()
        return super().__getstate__()

    def __setstate__(self, state):
        super().__setstate__(state)
        self._load_grads_from_state()

    def _store_grads_in_state(self):
        for group in self.param_groups:
            for param in group['params']:
                if isinstance(param, torch.Tensor) and param.grad is not None:
                    self.state.setdefault(param, {})['grad_state'] = param.grad

    def _load_grads_from_state(self):
        for (param, state) in self.state.items():
            if 'grad_state' in state:
                param.grad = state['grad_state']
            elif isinstance(param, torch.Tensor):
                param.grad = None


@torch.compile
def zeropower_via_newtonschulz5(G, steps=5):
    """
    From: https://github.com/KellerJordan/modded-nanogpt/blob/master/records/101724_DistributedMuon/22d24867-eb5a-4fcc-ae2c-263d0277dfd1.txt
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T

    # Ensure spectral norm is at most 1
    X = X / (X.norm() + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X
    
    if G.size(0) > G.size(1):
        X = X.T
    return X


@torch.compile(dynamic=False)
def zeropower_via_newtonschulz5_batched(G, steps=5):
    """``zeropower_via_newtonschulz5`` applied to each matrix of a (k, m, n) stack."""
    assert len(G.shape) == 3
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    transposed = G.size(-2) > G.size(-1)
    if transposed:
        X = X.mT

    # Per-matrix spectral norm bound, as X / (X.norm() + 1e-7) for one matrix.
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if transposed:
        X = X.mT
    return X


def _spectral_lmo_scale(backend, shape):
    d_out, d_in = shape
    scale = (d_out / d_in)**0.5 if backend.normalized else d_out**0.5
    if backend.max:
        scale = max(1, scale)
    return scale


def _lmo_all(norm_backend, tensors):
    """``norm_backend.lmo`` for each tensor; Spectral matrices batched by shape."""
    if not isinstance(norm_backend, Spectral):
        return [norm_backend.lmo(g) for g in tensors]
    out = [None] * len(tensors)
    buckets = {}
    for i, g in enumerate(tensors):
        if g.ndim == 2:
            buckets.setdefault((tuple(g.shape), g.device), []).append(i)
        else:
            out[i] = norm_backend.lmo(g)
    for (shape, _), idx in buckets.items():
        if len(idx) == 1:
            out[idx[0]] = norm_backend.lmo(tensors[idx[0]])
            continue
        X = zeropower_via_newtonschulz5_batched(
            torch.stack([tensors[i] for i in idx]), steps=norm_backend.steps)
        X *= _spectral_lmo_scale(norm_backend, shape)
        for j, i in enumerate(idx):
            out[i] = X[j]
    return out


def zeroth_power_via_svd(G):
   U, S, V = G.svd()
   return U @ V.T
