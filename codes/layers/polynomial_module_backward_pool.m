function [y, dU] = polynomial_module_backward_pool(x, dzdx, params)
U = params;
[h, w, c, n] = size(x);
dzdx = repmat(dzdx, [h, w, 1, 1]);

yM = zeros([h*w, c, n], 'like', dzdx);
dzdu = {zeros(size(U{1}), 'like', dzdx)};
dU = repmat(dzdu, size(U));

x = permute(x, [1,2,4,3]);
xM = reshape(x, h*w*n, c);

xU = cell(1, numel(U));
for d = 1:numel(U)
    xU{d} = xM * U{d};
end

dzdy = permute(dzdx, [1, 2, 4, 3]);
dzdy = reshape(dzdy, h*w*n, size(U{1}, 2));

for d = 1:numel(U)
    dIDX = 1:numel(U);
    dIDX(d) = [];
    Us = dzdy;
    for s = dIDX
        Us = Us.*xU{s};
    end
    dU{d} = xM'*Us;
    yMd = reshape(Us*U{d}', h*w, n, c);
    yM = yM + permute(yMd, [1, 3, 2]);
end

y = reshape(yM, h, w, c, n);
end
