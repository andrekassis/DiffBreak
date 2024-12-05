class ID:
    def __init__(
        self,
        model,
        eot_iters=1,
        targeted=False,
    ):
        self.model = model
        self.loss_fn = model.get_loss_fn()
        self.eot_iters = eot_iters
        assert targeted is False
        self.targeted = targeted

    def eval(self):
        self.model = self.model.eval()
        return self

    def to(self, device):
        self.model = self.model.to(device)
        return self

    def __call__(self, x, y, y_init):
        self.model.set_y_orig(y_init)
        _, successful_attack, single_succ = self.model.eval_attack(
            x.detach(),
            y,
            targeted=self.targeted,
        )
        if single_succ:
            single_found = True
            x_adv_single = x.detach()
        else:
            single_found = False

        return x, float(successful_attack), float(single_found)
