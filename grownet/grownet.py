import torch
import torch.nn as nn


class DynamicNet:
    def __init__(self, c0, lr, device, learnBoostRate=False, propagate_context=True):
        super(DynamicNet, self).__init__()
        self.models = []
        self.c0 = c0
        self.lr = lr
        self.device = device
        self.boost_rate = nn.Parameter(torch.tensor(lr, requires_grad=True, device=self.device))
        self.learnBoostRate = learnBoostRate
        self.propagate_context = propagate_context

    def __repr__(self):
        return str(self.models)

    def add(self, model):
        self.models.append(model)

    def pop(self):
        self.models.pop()

    def state_dict(self):
        state_dicts = []
        for m in self.models:
            state_dicts.append(m.state_dict())
        return state_dicts

    def load_state_dict(self, state_dicts, cfg, get_model_fn):
        for i in range(len(state_dicts)):
            model = get_model_fn(cfg, i)
            if model:
                model.to(self.device)
            self.models.append(model)
            self.models[i].load_state_dict(state_dicts[i])

    def parameters(self, recurse=True):
        params = []
        for m in self.models:
            params.extend(m.parameters())

        if self.learnBoostRate:
            params.append(self.boost_rate)
        return params

    def named_parameters(self, recurse=True):
        params = []
        for m in self.models:
            params.extend(m.named_parameters())

        if self.learnBoostRate:
            params.append(self.boost_rate)
        return params

    def zero_grad(self, set_to_none=False):
        for m in self.models:
            m.zero_grad()
        
        if self.learnBoostRate:
            self.boost_rate._grad = None  # Is this correct?

    def to_cuda(self):
        for m in self.models:
            m.cuda()

    def to(self, device):
        for m in self.models:
            m.to(device)

    def eval(self):
        for m in self.models:
            m.eval()

    def train(self):
        for m in self.models:
            m.train(True)

    def forward(self, x):
        if len(self.models) == 0:
            return None, self.c0
        middle_feat_cum = None
        preds = []
        with torch.no_grad():
            for m in self.models:
                middle_feat_cum, pred =  m(x, middle_feat_cum if self.propagate_context else None)
                preds.append(pred)
        prediction = sum(preds)
        # prediction = torch.stack(preds)
        # prediction = torch.cat((torch.sum(prediction[...,:3], dim=0),torch.sum(prediction[...,3:], dim=0)), dim=-1)
        return middle_feat_cum, self.c0 + self.boost_rate * prediction

    def forward_grad(self, x):
        if len(self.models) == 0:
            return None, self.c0
        # at least one model
        middle_feat_cum = None
        preds = []
        for m in self.models:
            middle_feat_cum, pred =  m(x, middle_feat_cum if self.propagate_context else None)
            preds.append(pred)
        prediction = sum(preds)
        # prediction = torch.mean(torch.stack(preds), dim=0)
        return middle_feat_cum, self.c0 + self.boost_rate * prediction

    def __call__(self, x):
        penultimate, out = self.forward(x)
        return penultimate, out
