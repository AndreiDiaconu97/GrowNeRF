from enum import Enum
import torch
import torch.nn as nn


class DynamicNet:
    def __init__(self, c0, lr):
        super(DynamicNet, self).__init__()
        self.models = []
        self.c0 = c0
        self.lr = lr
        self.boost_rate = nn.Parameter(torch.tensor(lr, requires_grad=True, device="cuda"))

    def __repr__(self):
        return str(self.models)

    def add(self, model):
        self.models.append(model)

    def state_dict(self):
        state_dicts = []
        for m in self.models:
            state_dicts.append(m.state_dict())
        return state_dicts

    # def load_state_dict(self, state_dicts, P):
    #     for i in range(len(state_dicts)):
    #         model = get_model(P, i)
    #         self.models.append(model)
    #         self.models[i].load_state_dict(state_dicts[i])

    def parameters(self, recurse=True):
        params = []
        for m in self.models:
            params.extend(m.parameters())

        params.append(self.boost_rate)
        return params

    def named_parameters(self, recurse=True):
        params = []
        for m in self.models:
            params.extend(m.named_parameters())

        params.append(self.boost_rate)
        return params

    def zero_grad(self, set_to_none=False):
        for m in self.models:
            m.zero_grad()
        # self.boost_rate._grad = None  # Is this correct?

    def to_cuda(self):
        for m in self.models:
            m.cuda()

    def to(self, device):
        for m in self.models:
            m.to(device)

    def to_eval(self):
        for m in self.models:
            m.eval()

    def to_train(self):
        for m in self.models:
            m.train(True)

    def forward(self, x):
        if len(self.models) == 0:
            return None, self.c0
        middle_feat_cum = None
        prediction = None
        with torch.no_grad():
            for m in self.models:
                if middle_feat_cum is None:
                    middle_feat_cum, prediction = m(x, middle_feat_cum)
                else:
                    middle_feat_cum, pred = m(x, middle_feat_cum)
                    prediction += pred
        return middle_feat_cum, self.c0 + self.boost_rate * prediction  # TODO: check if these parameters are necessary

    def forward_grad(self, x):
        if len(self.models) == 0:
            return None, self.c0
        # at least one model
        middle_feat_cum = None
        preds = []
        for m in self.models:
            middle_feat_cum, pred = m(x, middle_feat_cum)
            preds.append(pred)
        prediction = sum(preds)
        return middle_feat_cum, self.c0 + self.boost_rate * prediction

    def __call__(self, x):
        _, out = self.forward_grad(x)
        return out

    @classmethod
    def from_file(cls, path, builder):
        d = torch.load(path)
        net = DynamicNet(d['c0'], d['lr'])
        net.boost_rate = d['boost_rate']
        for stage, m in enumerate(d['models']):
            submod = builder(stage)
            submod.load_state_dict(m)
            net.add(submod)
        return net

    def to_file(self, path):
        models = [m.state_dict() for m in self.models]
        d = {'models': models, 'c0': self.c0, 'lr': self.lr, 'boost_rate': self.boost_rate}
        torch.save(d, path)
