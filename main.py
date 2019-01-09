from model import *
from trainer import Trainer


def main():
    Discrete_Vars = 10
    Continuous_Vars = 4
    Noise_Vars = 64
    OCT = False

    fe = FrontEnd(OCT)
    d = D()
    q = Q(Discrete_Vars, Continuous_Vars)
    g = G(Discrete_Vars + Continuous_Vars + Noise_Vars, OCT)

    for i in [fe, d, q, g]:
        i.cuda()
        i.apply(weights_init)

    trainer = Trainer(g, fe, d, q, Discrete_Vars, Continuous_Vars, Noise_Vars, 28, 28, 100, OCT)
    trainer.train()

if __name__ == "__main__":
    main()
