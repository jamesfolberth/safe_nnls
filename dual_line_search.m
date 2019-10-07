function [nu_hat] = dual_line_search(A, nu, nu_strict, varargin)
% Finds point on the line segment from nu to nu_strict that
% is closest to nu and is dual feasible (A'*nu >= 0).  nu_strict
% should be strictly dual feasible (A'*nu_strict > 0) for this
% to be guaranteed to work.

    % Get options
    p = inputParser;
    p.addParameter('feas_atol', 1e-14);
    p.addParameter('growth_factor', sqrt(10));
    
    p.parse(varargin{:});
    opt = p.Results;

    % Line search on segment from nu to nu_strict to find nu_hat
    % closest to nu but dual feasible.
    % nu_strict is assumed to be strictly feasible.
    % This uses a closed-form solution described in the notes.
    lambda = A'*nu;
    I = find(lambda < 0);
    if isempty(I)
        t = 0;
        nu_hat = nu;
    else
        lambda_strict = A'*nu_strict;
        if any(lambda_strict <= 0)
            error('nu_strict not strictly dual feasible.');
        end

        t = max(0, max(lambda(I)./(lambda(I) - lambda_strict(I))));
        nu_hat = (1-t)*nu + t*nu_strict;
    end
    
    % Ensure dual feasibility with some gap to account for roundoff
    scale = 1;
    while any(A'*nu_hat < opt.feas_atol)
        %fprintf('Rescaling to get stricter dual feasibility\n');

        t = t + scale*eps();
        nu_hat = (1-t)*nu + t*nu_strict;
        scale = scale*opt.growth_factor;
    end

end
