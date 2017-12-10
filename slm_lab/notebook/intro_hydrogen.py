'''
Example Hydrogen notebook
Use `lab` as your Hydrogen kernel and run below interactively
'''
from IPython.display import Latex
from slm_lab.lib import util, viz
import pandas as pd
import pydash as _

df = pd.DataFrame({
    'a': [0, 1, 2, 3, 4],
    'b': [0, 1, 4, 9, 16],
})

fig = viz.plot_area(df, ['a', 'b'])
fig = viz.plot_area(df, ['a'], y2_col=['b'])
fig = viz.plot_area(df, ['a', 'b'], stack=True)
fig = viz.plot_bar(df, ['b', 'a'])
fig = viz.plot_line(df, ['b', 'a'], save=False)
fig = viz.plot_line(df, ['a'], y2_col=['b'])
fig = viz.plot_scatter(df, ['b', 'a'])
fig = viz.plot_histogram(df, ['b'])

# pull plots to make multiple subplots
fig1 = viz.plot_area(df, ['a'], y2_col=['b'], draw=False)
fig2 = viz.plot_area(df, ['b'], draw=False)
fig = viz.tools.make_subplots(rows=3, cols=1, shared_xaxes=True)
fig.append_trace(fig1.data[0], 1, 1)
fig.append_trace(fig1.data[1], 2, 1)
fig.append_trace(fig2.data[0], 3, 1)
fig.layout['yaxis1'].update(fig1.layout['yaxis'])
fig.layout['yaxis2'].update(fig1.layout['yaxis2'])
fig.layout['yaxis1'].update(domain=[0.55, 1])
fig.layout['yaxis3'].update(fig2.layout['yaxis'])
fig.layout['yaxis3'].update(domain=[0, 0.45])
fig.layout.update(_.pick(fig1.layout, ['legend']))
fig.layout.update(title='total_rewards vs time', width=500, height=400)
viz.py.iplot(fig)


Latex(r'''\begin{eqnarray}
\nabla \times \vec{\mathbf{B}} -\, \frac1c\, \frac{\partial\vec{\mathbf{E}}}{\partial t} & = \frac{4\pi}{c}\vec{\mathbf{j}} \\
\nabla \cdot \vec{\mathbf{E}} & = 4 \pi \rho \\
\nabla \times \vec{\mathbf{E}}\, +\, \frac1c\, \frac{\partial\vec{\mathbf{B}}}{\partial t} & = \vec{\mathbf{0}} \\
\nabla \cdot \vec{\mathbf{B}} & = 0
\end{eqnarray}''')

Latex(r'''\begin{eqnarray}
\text{A policy is a function } \pi: S \rightarrow A \\
\text{Find a policy } \pi^* \text{that max. the cum. discounted reward } \sum_{t \geq 0}\gamma^t r_t \\
\pi^* = arg\max\limits_{\pi} E\big[\sum_{t\geq0}\gamma^t r_t|\pi\big] \\
\text{with } s_0 \sim p(s_0), a_t \sim \pi(\cdot|s_t),s_{t+1} \sim p(s_t, a_t) \\

\text{Value function: how good is a state?} \\
V^{\pi}(s) = E\big[\sum_{t\geq0}\gamma^t r_t|s_0=s,\pi \big] \\

\text{Q-value function: how good is a state-action pair?} \\
Q^{\pi}(s,a) = E\big[\sum_{t\geq0}\gamma^t r_t|s_0=s, a_0=a,\pi \big] \\
\end{eqnarray}''')

Latex(r'''\begin{eqnarray}
\pi^{*}(s) = \max\limits_{a} Q(s, a) \\
\end{eqnarray}''')


Latex(r'''\begin{eqnarray}
\text{Bellman equation for optimal policy: } \\
Q^* (s,a) = r(s, a) + \gamma E\big[V(s' | s, a)\big]  \\
Q^* (s,a) \approx r(s, a) + \gamma \ \max\limits_{a'} Q^* (s', a')|s, a  \\
\end{eqnarray}''')

Latex(r'''\begin{eqnarray}
\text{Neural net approx. with param } \theta: Q(s,a; \theta) \approx Q^* (s,a) \\

\text{Forward pass, loss function: } L_i(\theta_i) = E_{s,a \sim p(\cdot)}\big[ (y_i - Q(s,a; \theta_i))^2 \big] \\
\text{where } y_i = E_{s \sim \xi} \big[r + \gamma \ \max\limits_{a'} Q(s', a'; \theta_{i-1})|s, a \big] \\

\text{Backward pass: } \nabla_\theta L_i(\theta_i) = E_{s,a \sim p(\cdot), s \sim \xi}\big[ L_i(\theta_i) - \nabla_{\theta_i} Q(s,a; \theta_i) \big] \\
\end{eqnarray}''')


Latex(r'''\begin{eqnarray}
\text{Algorithm DQN} \\
\text{For i = 1 .... N:} \\
\quad \text{Gather data } {(s_i, a_i, r_i, s'_i)} \ \text{by acting in the environment using some policy} \\
\quad \text{for j = 1 ... K:} \\
\quad \quad \text{1. Calculate target values for each example} \\
\quad \quad y_i = r_i + \gamma \ \max\limits_{a'} Q(s'_i, a'; \theta_{i-1})|s_i, a_i \\
\quad \quad \text{2. Update network parameters, using MSE loss} \\
\quad \quad L_j(\theta) = \frac{1}{2} \sum_i || (y_i - Q(s_i,a_i; \theta_i)) ||^2 \\
\end{eqnarray}''')


Latex(r'''\begin{eqnarray}
\text{Define a class of parametrized policies, } \Pi = \{\pi_\theta, \theta \in \mathbb{R}^m\} \\
\text{For each } \pi_\theta \text{, define its value } J(\theta) = E\big[ \sum_{t\geq0}\gamma^t r_t|\pi_\theta \big] \\
\text{Find the optimal policy } \theta^* = arg\max\limits_\theta J(\theta) \\
\text{For trajectory } \tau = (s_0, a_0, r_0, s_1, \dots) \text{ and scalar score function } f(\tau) \\
J(\theta) = E_{\tau \sim p(\tau; \theta)} \big[ f(\tau) \big] = \int_{\tau} f(\tau) p(\tau; \theta) d\tau \\

\text{Gradient update for convergence (with some magic):} \\
\nabla_\theta J(\theta) = E_{\tau \sim p(\tau; \theta)} \big[ f(\tau) \nabla_\theta logp(\tau; \theta) \big] \\
\ \ \approx \sum_{t \geq 0} f(\tau) \nabla_\theta log \pi_\theta(a_t|s_t) \\
\end{eqnarray}''')

Latex(r'''\begin{eqnarray}
\text{Algorithm REINFORCE:} \\
\text{Initialize weights } \theta \text{, learning rate } \alpha \\
\text{for each episode (trajectory) } \tau = \{s_0, a_0, r_0, s_1, \cdots, r_T\} \sim \pi_\theta \\
\quad \text{for } t = 0 \text{ to } T \text{ do} \\
\quad \quad \theta \leftarrow \theta + \alpha \ f(\tau)_t \nabla_\theta log \pi_\theta(a_t|s_t) \\
\quad \text{end for} \\
\text{end for} \\
\end{eqnarray}''')

Latex(r'''\begin{eqnarray}
\text{Given } \nabla_\theta J(\theta) \ \approx \sum_{t \geq 0} f(\tau) \nabla_\theta log \pi_\theta(a_t|s_t), \text{improve baseline with: }\\
1.\ \text{reward as weightage } f(\tau) = \sum\limits_{t' \geq t} r_{t'} \\
2.\ \text{add discount factor } f(\tau) = \sum\limits_{t' \geq t} \gamma^{t'-t} r_{t'} \\
3.\ \text{introduce baseline } f(\tau) = \sum\limits_{t' \geq t} \gamma^{t'-t} r_{t'} - b(s_t) \\
4.\ \text{advantage function } f(\tau) = Q^\pi (s_t, a_t) - V^\pi (s_t) = A^\pi(s_t,a_t) \\
\nabla_\theta J(\theta) \approx \sum_{t \geq 0} \big( Q^{\pi_\theta} (s_t, a_t) - V^{\pi_\theta} (s_t) \big) \nabla_\theta log \pi_\theta(a_t|s_t)  \\

\nabla_\theta J(\theta) \approx \sum_{t \geq 0} \big( A^{\pi_\theta} (s_t, a_t) \big) \nabla_\theta log \pi_\theta(a_t|s_t)  \\
A^\pi(s_t,a_t) = Q^\pi (s_t, a_t) - V^\pi (s_t) \\
A^\pi(s_t,a_t) = r(s_t, a_t) + V^\pi (s'_t) - V^\pi (s_t) \\
\end{eqnarray}''')

Latex(r'''\begin{eqnarray}
Q^\pi (s_t, a_t) = r(s_t, a_t) + \sum\limits_{t' = t + 1} E_{\pi_\theta} [r(s'_t, a'_t) | s_t, a_t] \\
Q^\pi (s_t, a_t) = r(s_t, a_t) + E_{s_{t+1} \sim p(s_{t+1} | s_t, a_t)} \big[V^{\pi}(s_{t + 1})\big] \\
Q^\pi (s_t, a_t) \approx r(s_t, a_t) + V^{\pi}(s_{t + 1}) \\
\end{eqnarray}''')


Latex(r'''\begin{eqnarray}
\text{Bellman equation for optimal policy: } \\
Q^* (s,a) = r(s, a) + \gamma E\big[V(s' | s, a)\big]  \\
V (s_t) = r(s_t, a_t) + V(s'_t)  \\
\end{eqnarray}''')

Latex(r'''\begin{eqnarray}
\text{Algorithm Actor Critic} \\
\text{For i = 1 .... N:} \\
\quad \text{1. Gather data } {(s_i, a_i, r_i, s'_i)} \ \text{by acting in the environment using your policy} \\
\quad \text{2. Update V} \\
\quad \text{for j = 1 ... K:} \\
\quad \quad \text{Calculate target values for each example} \\
\quad \quad y_i = r_i + V(s'_i) \\
\quad \quad \text{Update network parameters, using MSE loss} \\
\quad \quad L_j(\theta) = \frac{1}{2} \sum_i || (y_i - V(s_i; \theta_i)) ||^2 \\
\quad \text{3. Evaluate A, } A^\pi(s_i,a_i) = r(s_i, a_i) + V^\pi (s'_i) - V^\pi (s_i)\\
\quad \text{4. Calculate gradient,} \nabla_{\phi}J(\phi) \approx \sum_i A^\pi_t(s_i, a_i) \nabla_\phi log \pi_\phi (a_i | s_i)\\
\quad \text{5. Use gradient to update parameters } \phi  \\
\end{eqnarray}''')

Latex(r'''\begin{eqnarray}
\text{Algorithm Actor-Critic:} \\
\text{Init. policy weights } \theta, \text{critic weights } \phi \\
\text{sample N episodes } \sim \pi_\theta \\
\Delta\theta \leftarrow 0 \\
\text{for } i = 1, \cdots, N: \\
\quad \text{for } t = 1, \cdots, T: \\
\quad \quad A^\pi_t = \sum\limits_{t' \geq t} \gamma^{t'-t}r_t - V_\phi (s_t) \\
\quad \quad \Delta\theta \leftarrow \Delta\theta + A^\pi_t \nabla_\theta log \pi_\theta (a_t, s_t) \\
\Delta\theta \leftarrow \frac{1}{N} \Delta\theta, \ \Delta\phi = \nabla_\phi \frac{1}{N} \sum\limits_{i=1}^{N} \sum\limits_{t=1}^{T} ||A_t||^2 \\
\theta \leftarrow \theta + \alpha \ \Delta\theta \\
\phi \leftarrow \phi + \beta \ \Delta\phi \\
\end{eqnarray}''')
