% A small example NNLS problem where we certify uniqueness
% This is the example givin in Section 6.1 of our paper.

clear all;
rng(271831)

m = 3; n = 5;
A = [1 6 -1 8 0;
     -2 7 1 8 2;
     3 1 4 1 -5;];
b = [-1;  2; 1;];

x_star = lsqnonneg(A,b);

%cvx_clear
%cvx_precision low
%cvx_begin quiet
%    variable x_hat(n)
%    minimize norm(A*x_hat-b,'fro')
%    x_hat >= 0
%cvx_end
%x_hat = [0; 0; 1; 0; 0.5];
%x_hat = [0; 0; .9344; 0; 0.5455];
x_hat = [0; 0; .9344; 0; 0.5455];

x_hat = zeros(n,1);
Lip = norm(A)^2;
% 206 are required for basic subproblem
% 199 are required for all inds dome subproblem
for i=1:250
    x_hat = max(0, x_hat - 1/Lip*A'*(A*x_hat-b));
end
%x_hat = [0; 0; .9282; 0; 0.5409];

nu_prime = A*x_hat - b;

%nu_strict = rand(m,1);
%nu_strict = max(nu_prime, 0);
%cvx_begin quiet
%    variable nu_strict(m)
%    variable t(1)
%    maximize t
%    A'*nu_strict >= t
%    sum(A'*nu_strict) == 1
%cvx_end
%nu_strict
%nu_strict = nu_strict / sum(nu_strict)
nu_strict = [0.56; 0.34; 0.1];

% Do the dual line search, as a cheap projection
nu_hat = dual_line_search(A, nu_prime, nu_strict);

% Or we could solve the orthogonal projection subproblem, which is expensive.
%cvx_begin quiet
%    variable nu_hat(m)
%    minimize norm(nu_hat - nu_prime)
%    A'*nu_hat >= 0
%cvx_end

x_hat
nu_hat

p_hat = 0.5*sum((A*x_hat-b).^2,1);
d_hat = -0.5*sum((nu_hat+b).^2,1) + 0.5*sum(b.^2,1);
gap = p_hat - d_hat

L = 1; % Lip constant of f(x) = 0.5*norm(x-b)^2; Lip(grad_f) = 1

% Use the basic SAFE feature elimination problem eqn. (10)
lower_bounds = feat_elim_dual_strong_concavity(A, nu_hat, L, gap)

% Or use the "all inds dome subproblem" eqn. (20)
%lower_bounds = NNLS_all_inds_dome_subproblem_mex(A, nu_hat, L*gap);

zero_inds = lower_bounds > 1e-14

% It's overdetermined
A_red = A(:,~zero_inds);

% It's full rank
s = svd(A_red)

if sum(zero_inds) >= n - m
    fprintf('NNLS solution is unique!\n');
else
    fprintf('Can''t conclude uniqueness\n');
end


dist2_bound = 2/s(end)^2*gap;
dist2 = norm(x_hat - x_star)^2;

fprintf('|x_hat - x_star|^2 <= %f\n', dist2_bound);
fprintf('|x_hat - x_star|^2 actual = %f\n', dist2);


% Now let's try the GLP lemma
% It works after 286 iterations.
% For randn matrices, the GLP lemma usually works first; roughtly 10x fewer
% iterations is typical.  So this is a "counterexample" for that, I suppose.
combs = combnk(1:n,m);
for i=1:size(combs,1)
    AI = A(:,combs(i,:));
    s = svd(AI);

    if s(end) < 1e-8
        error('A is NOT in GLP');
    end
end
fprintf('A is in GLP\n');

x_hat = zeros(n,1);
Lip = norm(A)^2;
it = 0;
while true
    nu_prime = A*x_hat - b;
    nu_hat = dual_line_search(A, nu_prime, nu_strict);
    d_hat = -0.5*sum((nu_hat+b).^2,1) + 0.5*sum(b.^2,1);

    if d_hat > 1e-14
        break
    end

    x_hat = max(0, x_hat - 1/Lip*A'*(A*x_hat-b));
    it = it + 1;
end
fprintf('GLP lemma worked at PGD it %d\n', it);

