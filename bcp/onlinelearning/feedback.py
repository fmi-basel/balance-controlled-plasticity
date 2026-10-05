import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Feedback projectors
# ---------------------------------------------------------------------------


class _GlobalFeedbackProjector:
    """Sums all controller dimensions into a single scalar broadcast to all inh. neurons."""

    def __init__(self, feedback_to_excitatory=False):
        self.feedback_to_excitatory = feedback_to_excitatory

    def __call__(self, ctrl, W_OUT):
        fb_inh = ctrl.sum()
        fb_exc = ctrl.sum() if self.feedback_to_excitatory else 0.0
        return fb_inh, fb_exc


class _RandomFeedbackProjector:
    """Fixed random feedback matrix, pre-projected onto the inhibitory population."""

    def __init__(
        self,
        W_FB_inh,
        W_FB_exc=None,
        nb_exc=0,
        only_disinhibitory=False,
    ):
        # W_FB_inh: (nb_outputs, nb_inh)
        self.W_FB_inh = W_FB_inh
        self.W_FB_exc = W_FB_exc
        self.nb_exc = nb_exc
        self.only_disinhibitory = only_disinhibitory

    def __call__(self, ctrl, W_OUT):
        fb_inh = jnp.dot(ctrl, self.W_FB_inh)
        if self.only_disinhibitory:
            fb_inh = jax.nn.relu(fb_inh)

        if self.W_FB_exc is not None:
            fb_exc = jnp.dot(ctrl, self.W_FB_exc)
        else:
            fb_exc = jnp.zeros(self.nb_exc)

        return fb_inh, fb_exc


class _StructuredFeedbackProjector:
    """Uses the learned readout W_OUT to project controller feedback (default)."""

    def __init__(
        self, M_I, M_E, only_disinhibitory=False, feedback_to_excitatory=False
    ):
        self.M_I = M_I
        self.M_E = M_E
        self.only_disinhibitory = only_disinhibitory
        self.feedback_to_excitatory = feedback_to_excitatory
        self.nb_exc = M_E.shape[0]

    def __call__(self, ctrl, W_OUT):
        W_FB_I = jnp.dot(W_OUT.T, self.M_I.T)
        fb_inh = jnp.dot(ctrl, W_FB_I)
        if self.only_disinhibitory:
            fb_inh = jax.nn.relu(fb_inh)

        if self.feedback_to_excitatory:
            W_FB_E = jnp.dot(W_OUT.T, self.M_E.T)
            fb_exc = jnp.dot(ctrl, W_FB_E)
        else:
            fb_exc = jnp.zeros(self.nb_exc)

        return fb_inh, fb_exc
