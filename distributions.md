# Supported distributions from the exponential family
    - Exponential family in canonical form:
        $$ p_{\pmb{eta}} \left( \pmb{x} \right) = h(\pmb{x}) e^{\pmb{\eta} \cdot \pmb{T}\left( \pmb{x} \right) - A\left( \pmb{eta} \right)} $$
    - Normal distribution:
             $$ \pmb{T} \left( x \right) = \left( x, x^2 \right)^T $$
             $$ A\left( \pmb{eta} \right) = - \frac{\eta_1^2}{ 4 \eta_2 } + \frac{1}{2} \ln \left| \frac{1}{2\eta_2} \right|$$
    - Gamma
            $$ \pmb{T}\left( x \right) = \left( \ln x, x \right)^T $$
            $$ A\left( \pmb{eta} \right) = \ln \Gamma \left( \eta_1 + 1 \right) - \left(\eta_1 + 1 \right) \ln \left( - \eta_2 \right) $$
    - Dirichlet
            $$ \pmb{T}\left( \pmb{x} \right) = \left( ln x_i | \forall i \in \left\{1, \ldots, k \right} \right)^T $$
            $$ A\left( \pmb{eta} \right) = \sum_{i=1}^k \ln \Gamma \left( \eta_i + 1 \right) - \ln \Gamma \left( \sum_{i=1}^k \eta_i + k \right)$$
    - Categorical
            $$ \pmb{T}\left( x \right) = \left( [x == i] | \forall i \in \left\{1, \ldots, k \right} \right)^T $$
            $$ A\left( \pmb{eta} \right) = \ln \left( \sum_{i=1}^k e^{\eta_i} \right) = \ln \left( 1 + \sum_{i=1}^{k-1} e^{\eta_i} \right) $$
