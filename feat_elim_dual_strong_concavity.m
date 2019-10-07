function [lower_bounds] = feat_elim_dual_strong_concavity(A, nu_hat, L, gap)
% Solves the "basic" feature elimination subproblem that uses
% the strong concavity of the dual problem.  This assumes A is
% full rank, L > 0 is the Lipshitz constant of the primal gradient
% (without A), and gap > 0 is the duality gap given by the pair of points
% x_hat and nu_hat.
    
    [m,n] = size(A);
    n_rhs = size(nu_hat,2);
    if m ~= size(nu_hat,1)
        error('incompatible sizes of A and nu_hat');
    end

    if size(gap,1) ~= 1 && size(gap,2) ~= n_rhs
        error('gap should be a scalar/row vector');
    end
    
    % Solve
    %   min <a_i,nu>
    %   s.t. ||nu - nu0||^2 <= 2*gap_bound
    %------------------------------
    dot_anu = A'*nu_hat;
    norm_a = sqrt(sum(A.^2,1));
    lower_bounds = dot_anu - norm_a'*sqrt(2*L*gap);

end
