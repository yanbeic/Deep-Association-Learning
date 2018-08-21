function cmc = compute_CMC(gallery_index, index)

cmc = zeros(length(index), 1);

for n = 1:length(index) 
    if ~isempty(find(gallery_index == index(n), 1)) 
        cmc(n:end) = 1;
    end
end 

end
