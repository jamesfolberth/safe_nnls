% Plot the distance to the solution as a function of the number of
% iterations of projected gradient descent.

clear all;

rng(271828);
m = 50; n = 100;

% find an NNLS problem with a unique solution
fprintf('Finding an NNLS problem with unique solution\n');
while true
    % A is in GLP w/p 1, so we can use Slawski & Hein Lemma to show uniqueness
    A = randn(m,n);
    b = randn(m,1);

    x_star = lsqnonneg(A,b);
    p_star = 0.5*norm(A*x_star - b)^2; 

    if p_star > 1e-8
        fprintf('Found one: p_star = %f\n', p_star);
        break
    end
end

% Find a strictly dual feasible point
fprintf('Finding a strictly dual feasible point\n');
cvx_precision best
cvx_begin quiet
    variable nu_strict(m)
    variable t(1)
    maximize t
    A'*nu_strict >= t
    sum(A'*nu_strict) == 1
cvx_end
nu_strict = nu_strict/norm(nu_strict,1);

% Now run PGD/FISTA
its = 0:7500;
n_its = numel(its);
t = 1/norm(A)^2; % constant step size 1/L

x_hat = zeros(n,1);
x_hat_old = zeros(n,1);
y = zeros(n,1);
tk = 1;
tk_old = tk;

gaps = zeros(n_its,1);
gaps_orth = zeros(n_its,1);
num_elim = zeros(n_its,1);
num_elim_orth = zeros(n_its,1);
num_elim2 = zeros(n_its,1);
dist_bounds = nan(n_its,1);
dist_bounds_star = nan(n_its,1);
dist_bounds_orth = nan(n_its,1);
dist_bounds_orth_star = nan(n_its,1);
dist_bounds2 = nan(n_its,1);
dist_bounds2_star = nan(n_its,1);
dist_actuals = zeros(n_its,1);
glp_lemma_first_it = nan;

for i=1:n_its
    fprintf('\rPGD iteration %d', its(i));
    % Find dual feasible point
    nu_prime = A*x_hat - b;
    nu_hat = dual_line_search(A, nu_prime, nu_strict);
    
    % Find the orthogonal projection
    %cvx_precision best
    %cvx_begin quiet
    %    variable nu_orth(m)
    %    minimize norm(nu_orth - nu_prime)
    %    A'*nu_orth >= 0
    %cvx_end
    opt = optimoptions('quadprog');
    opt.Display = 'none';
    opt.ConstraintTolerance = 1e-14;
    opt.OptimalityTolerance = 1e-14;
    opt.StepTolerance = 1e-14;
    nu_orth = quadprog(eye(m), -nu_prime, -A', zeros(n,1), [], [], [], [], nu_hat, opt);
    
    % Compute duality gap
    p_hat = 0.5*sum((A*x_hat-b).^2,1);
    d_hat = -0.5*sum((nu_hat+b).^2,1) + 0.5*sum(b.^2,1);
    gap = p_hat - d_hat;
    gaps(i) = gap;

    d_orth = -0.5*sum((nu_orth+b).^2,1) + 0.5*sum(b.^2,1);
    gap_orth = p_hat - d_orth;
    gaps_orth(i) = gap_orth;

    L = 1; % Lip constant of f(x) = 0.5*norm(x-b)^2; Lip(grad_f) = 1
    
    % Do basic feature elimination subproblem
    lower_bounds = feat_elim_dual_strong_concavity(A, nu_hat, L, gap);
    zero_inds = lower_bounds > 1e-14;
    num_elim(i) = sum(zero_inds);

    lower_bounds_orth = feat_elim_dual_strong_concavity(A, nu_orth, L, gap_orth);
    zero_inds_orth = lower_bounds_orth > 1e-14;
    num_elim_orth(i) = sum(zero_inds_orth);
 
    % Do strong concavity + singleton dual feasibility
    lower_bounds2 = NNLS_all_inds_dome_subproblem_mex(A, nu_hat, L*gap);
    zero_inds2 = lower_bounds2 > 1e-14;
    num_elim2(i) = sum(zero_inds2);
    
    fprintf('  num_elim=%03d  num_elim2=%03d', num_elim(i), num_elim2(i));
    
    % Can we certify uniqueness?
    if num_elim(i) >= n - m
        % Yup!
        % It's overdetermined
        A_red = A(:,~zero_inds);

        % It's full rank (implied by A being in GLP)
        s = svd(A_red);
        
        % Bound the distance to the solution
        dist_bounds(i) = sqrt(2*gap)/s(end);

        % For comparison, bound the distance to the solution using
        % the true value of f(A*x_hat) - f(A*x_star)
        dist_bounds_star(i) = sqrt(2*(p_hat - p_star))/s(end);
    end

    if num_elim_orth(i) >= n - m
        A_red = A(:,~zero_inds_orth);
        s = svd(A_red);
        dist_bounds_orth(i) = sqrt(2*gap_orth)/s(end);
        dist_bounds_orth_star(i) = sqrt(2*(p_hat - p_star))/s(end);
    end

    if num_elim2(i) >= n - m
        A_red = A(:,~zero_inds2);
        s = svd(A_red);
        dist_bounds2(i) = sqrt(2*gap)/s(end);
        dist_bounds2_star(i) = sqrt(2*(p_hat - p_star))/s(end);
    end
 
    % True distance to the solution
    dist_actuals(i) = norm(x_hat - x_star);

    % Can we certify uniqueness via GLP lemma?
    if isnan(glp_lemma_first_it) && d_hat > 1e-14;
        glp_lemma_first_it = its(i);
    end
    
    % PGD step - a descent method
    x_hat = max(0, x_hat - t*A'*(A*x_hat-b));

    % FISTA step - not a descent method
    %x_hat_old = x_hat;
    %x_hat = max(0, y - t*A'*(A*y-b));
    %tk_old = tk;
    %tk = 0.5*(1 + sqrt(1 + 4*tk^2));
    %y = x_hat + (tk_old-1)/tk * (x_hat - x_hat_old);
end
fprintf('\n');


figure(1);
set(gcf, 'Position',  [0, 0, 900, 250])
set(gcf, 'DefaultTextInterpreter','LaTeX',...
         'DefaultAxesTickLabelInterpreter', 'LaTeX',...
         'DefaultLegendInterpreter', 'LaTeX');
clf;
subplot(131);
hold on
plot(its, num_elim, 'LineWidth', 2);%, 'HandleVisibility', 'off');
ax = gca;
ax.ColorOrderIndex = 1;
plot(its, num_elim_orth, '-.', 'LineWidth', 2);%, 'HandleVisibility', 'off');
%plot(its, num_elim2, 'LineWidth', 2, 'HandleVisibility', 'off');
num_inactive_star = n - sum(x_star > 1e-14);
plot([its(1); its(end)], [num_inactive_star; num_inactive_star], 'k:', 'LineWidth', 2);
I = find(num_elim >= n - m);
if ~isempty(I)
    plot([I(1); I(1)], [0 n-m], 'k--', 'LineWidth', 2);
    plot([0; I(1)], [n-m n-m], 'k--', 'LineWidth', 2);
end
hold off
legend('SAFE', 'SAFE orth. proj.', 'Total possible', 'For SAFE uniqueness', 'Location', 'NorthWest');
xlabel('Iteration');
ylabel('Number Eliminated');
xlim([0 its(end)]);
ylim([0 n]);

subplot(132);
hold on
plot(its, gaps, 'LineWidth', 2);%, 'HandleVisibility', 'off');
ax = gca;
ax.ColorOrderIndex = 1;
plot(its, gaps_orth, '-.', 'LineWidth', 2);%, 'HandleVisibility', 'off');
if ~isnan(glp_lemma_first_it)
    gap = gaps(max(1,glp_lemma_first_it));
    plot([its(1); its(end)], [gap; gap], 'k--', 'LineWidth', 2);
end
hold off
lg = legend('SAFE', 'SAFE orth. proj.', 'For Lemma 6.1', 'Location', 'SouthWest');
set(lg, 'Color', 'None'); % transparent background
xlabel('Iteration');
ylabel('Duality Gap');
xlim([0 its(end)]);
set(gca, 'yscale', 'log');

subplot(133)
hold on
plot(its, dist_bounds, 'LineWidth', 2);
ax = gca;
ax.ColorOrderIndex = 1;
plot(its, dist_bounds_orth, '-.', 'LineWidth', 2);
ax.ColorOrderIndex = 1;
plot(its, dist_bounds_star, '--', 'LineWidth', 2);
plot(its, dist_actuals, 'k:', 'LineWidth', 2);
hold off
lg = legend('SAFE bound', 'SAFE orth. proj. bound', 'SAFE bound$\phantom{}^\ast$', 'True distance', 'Location', 'SouthWest');
set(lg, 'Color', 'None'); % transparent background
xlabel('Iteration');
ylabel('Distance to Solution')
xlim([0 its(end)]);
ylim([1e-7 1e1]);
set(gca, 'yscale', 'log');


