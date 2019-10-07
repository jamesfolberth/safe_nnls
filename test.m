function [] = test()
    
    %test_dual_line_search()
    test_feat_elim_dual_strong_concavity()

end

function [] = test_dual_line_search()

    A = rand(3,5);
    nu = randn(3,1);
    nu_strict = rand(3,1);

    nu_hat = dual_line_search(A, nu, nu_strict);

    A'*nu
    A'*nu_hat

end

function [] = test_feat_elim_dual_strong_concavity()
    
    rng(271831)
    
    m = 3; n = 5; n_rhs = 2;
    A = rand(m,n);
    b = randn(m,n_rhs);
    
    cvx_clear
    cvx_precision low
    cvx_begin quiet
        variable x_hat(n,n_rhs)
        minimize norm(A*x_hat-b,'fro')
        x_hat >= 0
    cvx_end

    nu_prime = A*x_hat - b;
    %nu_strict = rand(m,n_rhs);
    nu_strict = max(nu_prime, 0);
    nu_hat = dual_line_search(A, nu_prime, nu_strict);

    x_hat
    nu_hat

    p_hat = 0.5*sum((A*x_hat-b).^2,1);
    d_hat = -0.5*sum((nu_hat+b).^2,1) + 0.5*sum(b.^2,1);
    gap = p_hat - d_hat
    
    L = 1;

    lower_bounds = feat_elim_dual_strong_concavity(A, nu_hat, L, gap)
    zero_inds = lower_bounds > 1e-14

end

