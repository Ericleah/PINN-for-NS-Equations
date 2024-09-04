import random

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from torch.autograd import Variable
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
filepath_to_save_mode = 'KT.pt'

R = np.linspace(1, 2, 2000).reshape(-1, 1)
Rleft = np.ones((1000, 1)).reshape(-1, 1)
Rright = 2 * np.ones((1000, 1)).reshape(-1, 1)

Pr = 6.2
Rd = Nr = 0.5
Ha = 0.5
Tr = 1.5
Phi = 0.1
Gr = 5
Qt = Qe = 0.01
gamma = 1
Nb=Nt=0.2
L1=0.1
Bt=Bc=0.2
cosa=0.5

R = Variable(torch.from_numpy(R).float(), requires_grad=True).to(device)
xleft = Variable(torch.from_numpy(Rleft).float(), requires_grad=True).to(device)
xright = Variable(torch.from_numpy(Rright).float(), requires_grad=True).to(device)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden_layer1 = nn.Linear(1, 64)
        self.hidden_layer2 = nn.Linear(64, 64)
        self.hidden_layer3 = nn.Linear(64, 64)
        self.hidden_layer4 = nn.Linear(64, 64)
        self.output_layer = nn.Linear(64, 3)

    def forward(self, x):
        layer1_out = torch.tanh(self.hidden_layer1(x))
        layer2_out = torch.tanh(self.hidden_layer2(layer1_out))
        layer3_out = torch.tanh(self.hidden_layer3(layer2_out))
        layer4_out = torch.tanh(self.hidden_layer4(layer3_out))
        output = self.output_layer(layer4_out)
        return output

    def loss1(self, x):
        u = net(x)[:, 0].reshape(-1, 1)
        theta = net(x)[:, 1].reshape(-1, 1)
        phi = net(x)[:, 2].reshape(-1, 1)

        du_dr = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        duu_drr = torch.autograd.grad(du_dr.sum(), x, create_graph=True)[0]
        dtheta_dr = torch.autograd.grad(theta.sum(), x, create_graph=True)[0]
        dthetatheta_drr = torch.autograd.grad(dtheta_dr.sum(), x, create_graph=True)[0]
        dphi_dr = torch.autograd.grad(phi.sum(), x, create_graph=True)[0]
        dphiphi_drr = torch.autograd.grad(dphi_dr.sum(), x, create_graph=True)[0]

        term1 = duu_drr + (1/x)*du_dr - Ha*u - Phi + (Gr*theta-Nr*phi)*cosa
        f1 = Tr-1
        f11 = (f1*theta+1)**3
        f111 = dthetatheta_drr + (1/x)*dtheta_dr
        F1 = (1+Rd*f11)*f111
        F2 = Pr*Nb*dtheta_dr*dphi_dr
        f3 = Qe*torch.exp(-gamma*x)
        F3 = Pr*(Qt*theta+f3)
        f4 = 3*Rd*f1
        f44 = (f1*theta+1)**2
        F4 = (f4*f44+Nt*Pr)*(dtheta_dr**2)

        # term2 = (1+Rd*((Tr-1)*theta+1)**3)*(dthetatheta_drr+(1/x)*dtheta_dr) + Pr*Nb*dtheta_dr*dphi_dr \
        #         + Pr*(Qt*theta + Qe*torch.exp(-gamma*x)) + (3*Rd*(Tr-1)*(((Tr-1)*theta+1)**2)+Nt*Pr)*(dtheta_dr**2)

        term2 = F1 + F2 + F3 + F4
        term3 = dphiphi_drr + (1/x)*dphi_dr + (Nt/Nb)*(dthetatheta_drr + (1/x)*dtheta_dr)

        return lossa(term1, 0 * term1) + lossa(term2, 0 * term2) + lossa(term3, 0 * term3)

    def lossbc1(self, x):
        u = net(x)[:, 0].reshape(-1, 1)
        theta = net(x)[:, 1].reshape(-1, 1)
        phi = net(x)[:, 2].reshape(-1, 1)

        du_dr = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        dtheta_dr = torch.autograd.grad(theta.sum(), x, create_graph=True)[0]
        dphi_dr = torch.autograd.grad(phi.sum(), x, create_graph=True)[0]
        f1 = u - L1*du_dr
        f2 = Bt*theta - dtheta_dr
        f3 = Bc*phi - dphi_dr

        loss1 = lossa(f1,0*f1) + lossa(f2, 0*f2) + lossa(f3, 0*f3)
        return loss1

    def lossbc2(self, x):
        u = net(x)[:, 0].reshape(-1, 1)
        theta = net(x)[:, 1].reshape(-1, 1)
        phi = net(x)[:, 2].reshape(-1, 1)

        du_dr = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        dtheta_dr = torch.autograd.grad(theta.sum(), x, create_graph=True)[0]
        dphi_dr = torch.autograd.grad(phi.sum(), x, create_graph=True)[0]

        f1 = u + L1*du_dr
        f2 = Bt*(1-theta) - dtheta_dr
        f3 = Bc*(1-phi) - dphi_dr

        loss1 = lossa(f1, 0 * f1) + lossa(f2, 0 * f2) + lossa(f3, 0 * f3)
        return loss1

lossa = torch.nn.MSELoss()
net = Net().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=5e-4)

R_plot = torch.linspace(1,2,100).reshape(-1, 1)
# 遍历不同的Ha,分别训练和绘图
Ha_vals = [0.1, 0.2, 0, 0.5, 1.0, 1.5, 2.0]

# Ha = 0
# for epoch in range(50000):
#     optimizer.zero_grad()
#     loss = net.loss1(R) + net.lossbc1(xleft) + net.lossbc2(xright)
#     loss.backward()
#     optimizer.step()
#
#     if epoch % 100 == 0:
#         print(f'Epoch {epoch}, Loss: {loss.item()}')
# torch.save(net.state_dict(), f'model_fitted.pt')

for Ha in Ha_vals:
    for epoch in range(5000):
        optimizer.zero_grad()
        loss = net.loss1(R) + net.lossbc1(xleft) + net.lossbc2(xright)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')
    torch.save(net.state_dict(), f'model_{Ha}1.pt')
#
Ha_vals = [0, 0.5, 1.0, 1.5, 2.0]
for Ha in Ha_vals:
    net.load_state_dict(torch.load(f'model_{Ha}.pt'))

    R = torch.linspace(1, 2, 100)[:, None]
    with torch.no_grad():
        u = net(R)[:, 0]

    plt.plot(R, u, label=f'Ha={Ha}')

plt.legend()
plt.xlabel('R')
plt.ylabel('U(R)')
plt.title('Network Output for Different Ha')
plt.show()
plt.show()

# epoch=0
# while True:
#     optimizer.zero_grad()
#     loss = net.loss1(R) + net.lossbc1(xleft) + net.lossbc2(xright)
#     loss.backward()
#     optimizer.step()
#
#     if loss.item() < 1e-7:  # loss less than 10e-8
#         break
#
#     epoch += 1
#     if epoch % 100 == 0:
#         print(f'Epoch {epoch}, Loss: {loss.item()}')
#     max_iterations = 50000
#     if epoch >= max_iterations:
#         break
#
#     theta1 = net(xleft)[:, 1].reshape(-1, 1)
#     dtheta_dr1 = torch.autograd.grad(theta1.sum(), xleft, create_graph=True)[0]
#     theta2 = net(xright)[:, 1].reshape(-1, 1)
#     dtheta_dr2 = torch.autograd.grad(theta2.sum(), xright, create_graph=True)[0]
#
#
# result1 = -dtheta_dr1.detach().numpy()[0]
# result2 = -dtheta_dr2.detach().numpy()[0]
# print(result1)
# print(result2)

