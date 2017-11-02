function y = polynomial_module_forward_pool(x, params)
  U = params;
  pm_dim = size(U{1}, 2);
  [h, w, c, n] = size(x);

  x = permute(x, [1, 2, 4, 3]);
  xM = reshape(x, h*w*n, c);
  
  polyM = 1;
  for d = 1:numel(U)
    polyM = polyM .* (xM * U{d});
  end
  polyM = reshape(polyM, h, w, n, pm_dim);

  y = permute(polyM, [1, 2, 4, 3]);
  y = sum(sum(y, 1), 2);
end
