from model import *
from trainer import Trainer


def main():
    Discrete_Vars = 2
    Continuous_Vars = 20
    Noise_Vars = 16
    OCT = True

    fe = FrontEnd(OCT)
    d = D()
    q = Q(Discrete_Vars, Continuous_Vars)
    g = G(Discrete_Vars + Continuous_Vars + Noise_Vars, OCT)

    for i in [fe, d, q, g]:
        i.cuda()
        i.apply(weights_init)

    trainer = Trainer(g, fe, d, q, Discrete_Vars, Continuous_Vars, Noise_Vars, 512, 512, 2, OCT)
    trainer.train()

if __name__ == "__main__":
    main()