### A Pluto.jl notebook ###
# v0.19.0

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ cd9800b0-bb3a-11ec-1091-7985413e1e8e
begin
	using Pkg
	Pkg.add(PackageSpec(name="Glib_jll",version="2.59"))
	Pkg.add(["Images"
	,"ImageShow"
	,"ImageIO", "VideoIO","ImageCore"
	,"ImageFeatures", "ImageDraw", "CoordinateTransformations", "Statistics"
	,"Rotations","ProgressMeter","PlutoUI","Plots","PlotlyJS","DataFrames","ImageSegmentation","ImageEdgeDetection","LsqFit","JuMP"])
	using Images, ImageCore
	using Plots, DataFrames
	using ImageSegmentation, ImageEdgeDetection
	using ImageEdgeDetection: Percentile
	using LinearAlgebra,JuMP
	using Statistics
	using LsqFit
	using ImageIO, VideoIO
	using ProgressMeter,PlutoUI
	using ImageFeatures, ImageDraw, CoordinateTransformations, Rotations
	plotlyjs()
end

# ╔═╡ 42635499-73a1-4a2b-81be-b73cacb8db7c
video = VideoIO.load(joinpath("/home/randyt","2022-04-13_22-18-58.mkv"))

# ╔═╡ fdfd2b19-a817-4a9d-910e-3a17229f7907
@bind index Slider(1:size(video)[1],show_value=true)

# ╔═╡ 0ed7820d-ec75-4a47-9679-e69cc2c7cd17
sample = video[index]

# ╔═╡ 325d5ac4-e710-4881-8279-6c569ba76509
Gray.(sample)

# ╔═╡ 771124c2-7e8a-41f9-bfbf-987121481910
begin
	r = Float64.(view(channelview(sample),1,:,:))
	g = Float64.(view(channelview(sample),2,:,:))
	b = Float64.(view(channelview(sample),3,:,:))
end

# ╔═╡ f1b5f625-487c-4955-a37c-afab56b03afd
scatter(vec(r),vec(g),vec(b),markersize=0.1)

# ╔═╡ 2c14fc4d-a3cb-4bc1-a802-918c8fb11f3c
colors = collect.(zip(map(vec,(r,g,b))...))

# ╔═╡ f9e95fa5-8120-4b6c-bd29-9c91a5fc64d4
mean(colors)

# ╔═╡ fa6eed51-f831-4d0a-8ac2-78211170cf80
minimum(colors)

# ╔═╡ f05bc3ce-634c-4dca-b8d5-8c6cc0b7e860
thresh = (mean(colors) - minimum(colors)) .* 0.5 + minimum(colors)

# ╔═╡ 96db2505-4134-4d23-96cc-81ebb2247564
mean(reshape(Float64.(channelview(sample)),(3,prod(size(sample)))),dims=2)

# ╔═╡ 8200374b-4ef4-452b-bf8b-862775bbd00e
function cmp(x,y,fn)
	channelsize = size(channelview(x))[1]

	channels = [view(channelview(x),i,:,:) for i=1:channelsize]
	th = [broadcast(fn,channels[i],y[i]) for i=1:channelsize]
	reduce(.&,th)

end

# ╔═╡ 83bce8f4-459b-400c-9be7-fa985169acf3
begin
	function Base.:<(x::T, y::U) where {T<:Real, U<:FixedPoint}
	    if x >= round(T, typemax(y), RoundUp)
	        return false
	    elseif x <= round(T, typemin(y), RoundDown)
	        return true
	    end
	    _x, _y = promote(x,y)
	    return <(_x.i, _y.i)
	end
	Base.:<(x::U, y::T) where {U<:Fixed, T<:Real} = !(y < x)
end

# ╔═╡ 51edc0ff-85e2-441b-9ad8-9f3902c3250f
begin
	# rotate direction clocwise
	function clockwise(dir)
	    return (dir)%8 + 1
	end
	
	# rotate direction counterclocwise
	function counterclockwise(dir)
	    return (dir+6)%8 + 1
	end
	
	# move from current pixel to next in given direction
	function move(pixel, image, dir, dir_delta)
	    newp = pixel + dir_delta[dir]
	    height, width = size(image)
	    if (0 < newp[1] <= height) &&  (0 < newp[2] <= width)
	        if image[newp]!=0
	            return newp
	        end
	    end
	    return CartesianIndex(0, 0)
	end
	
	# finds direction between two given pixels
	function from_to(from, to, dir_delta)
	    delta = to-from
	    return findall(x->x == delta, dir_delta)[1]
	end
	
	
	
	function detect_move(image, p0, p2, nbd, border, done, dir_delta)
	    dir = from_to(p0, p2, dir_delta)
	    moved = clockwise(dir)
	    p1 = CartesianIndex(0, 0)
	    while moved != dir ## 3.1
	        newp = move(p0, image, moved, dir_delta)
	        if newp[1]!=0
	            p1 = newp
	            break
	        end
	        moved = clockwise(moved)
	    end
	
	    if p1 == CartesianIndex(0, 0)
	        return
	    end
	
	    p2 = p1 ## 3.2
	    p3 = p0 ## 3.2
	    done .= false
	    while true
	        dir = from_to(p3, p2, dir_delta)
	        moved = counterclockwise(dir)
	        p4 = CartesianIndex(0, 0)
	        done .= false
	        while true ## 3.3
	            p4 = move(p3, image, moved, dir_delta)
	            if p4[1] != 0
	                break
	            end
	            done[moved] = true
	            moved = counterclockwise(moved)
	        end
	        push!(border, p3) ## 3.4
	        if p3[1] == size(image, 1) || done[3]
	            image[p3] = -nbd
	        elseif image[p3] == 1
	            image[p3] = nbd
	        end
	
	        if (p4 == p0 && p3 == p1) ## 3.5
	            break
	        end
	        p2 = p3
	        p3 = p4
	    end
	end
	
	
	function find_contours(image)
	    nbd = 1
	    lnbd = 1
	    image = Float64.(image)
	    contour_list =  Vector{typeof(CartesianIndex[])}()
	    done = [false, false, false, false, false, false, false, false]
	
	    # Clockwise Moore neighborhood.
	    dir_delta = [CartesianIndex(-1, 0) , CartesianIndex(-1, 1), CartesianIndex(0, 1), CartesianIndex(1, 1), CartesianIndex(1, 0), CartesianIndex(1, -1), CartesianIndex(0, -1), CartesianIndex(-1,-1)]
	
	    height, width = size(image)
	
	    for i=1:height
	        lnbd = 1
	        for j=1:width
	            fji = image[i, j]
	            is_outer = (image[i, j] == 1 && (j == 1 || image[i, j-1] == 0)) ## 1 (a)
	            is_hole = (image[i, j] >= 1 && (j == width || image[i, j+1] == 0))
	
	            if is_outer || is_hole
	                # 2
	                border = CartesianIndex[]
	
	                from = CartesianIndex(i, j)
	
	                if is_outer
	                    nbd += 1
	                    from -= CartesianIndex(0, 1)
	
	                else
	                    nbd += 1
	                    if fji > 1
	                        lnbd = fji
	                    end
	                    from += CartesianIndex(0, 1)
	                end
	
	                p0 = CartesianIndex(i,j)
	                detect_move(image, p0, from, nbd, border, done, dir_delta) ## 3
	                if isempty(border) ##TODO
	                    push!(border, p0)
	                    image[p0] = -nbd
	                end
	                push!(contour_list, border)
	            end
	            if fji != 0 && fji != 1
	                lnbd = abs(fji)
	            end
	
	        end
	    end
	
	    return contour_list
	
	
	end
	
	# a contour is a vector of 2 int arrays
	function draw_contour(image, color, contour)
	    for ind in contour
	        image[ind] = color
	    end
	end
	function draw_contours(image, color, contours)
	    for cnt in contours
	        draw_contour(image, color, cnt)
	    end
	end
end

# ╔═╡ 537ec3f4-4f90-41f1-88df-41517b925ec2
begin
	struct HaarSurroundFeature
		r_inner
		r_outer
		val_inner
		val_outer
	end
	function HaarSurroundFeature(r_inner,r_outer)
			count_inner = r_inner*r_inner
			count_outer = r_outer*r_outer - count_inner
			return HaarSurroundFeature(r_inner,r_outer,1.0/count_inner,1.0/count_outer)
	end
	HaarSurroundFeature(radius) = HaarSurroundFeature(radius,radius*3)
end

# ╔═╡ 63b77666-ea9a-40a8-a66a-fd5339cd8ce5
function HaarKernel(properties::HaarSurroundFeature) 
	r1 = properties.r_inner
	pad = Integer((properties.r_outer - r1)/2)
	vi = properties.val_inner
	vo = properties.val_outer

		padarray(fill(-vi,(r1,r1)),Fill(vo,(pad,pad)))
end

# ╔═╡ 60955ee9-3e37-49bb-92ba-0b2dc7c7d1c2
HaarKernel(HaarSurroundFeature(2,4))

# ╔═╡ 13db6529-c736-4524-9396-b3b461a380fe
Gray.(HaarKernel(HaarSurroundFeature(2,4)))

# ╔═╡ f4fdd861-11af-447c-bb02-0d4f76dbb4a6
Gray.(Gray.(HaarKernel(HaarSurroundFeature(2,4))) .>0)

# ╔═╡ 93dba41e-4eb7-436e-93d2-b4d4be23dfdb
begin
	rstep = 2
	minRadius,maxRadius = (30,120)
	PercentageInliers = .3
	CannyMinThreshold = 95
    CannyMaxThreshold = 100
end

# ╔═╡ bcd7d54d-7506-48e5-8e1b-08955143c8ec
begin
	maxResponse = -Inf
	response = []
	haarRadius = 0

	
	for radius in minRadius:rstep:maxRadius
		tmp = imfilter(Gray.(sample),HaarKernel(HaarSurroundFeature(radius)))
		if maximum(tmp) > maxResponse
			maxResponse = maximum(tmp)
			response = tmp
			haarRadius = radius
		end
		
	end
	center = argmax(response)
	haarRadius = haarRadius*2^0.5
end

# ╔═╡ 8aaeae0e-1e11-474c-8926-178840367d16
Tuple(center)

# ╔═╡ c523fcae-6627-493a-951b-7b8e585fe52f
haarRadius

# ╔═╡ 598c5201-2409-43fd-8673-5a827db79b04
float64(sample[argmax(response)])

# ╔═╡ 2955fbde-5718-4ed1-ade5-e2466318bdf4
Gray.(response.>quantile(vec(response),.95))

# ╔═╡ 8b029d5b-becf-4de7-a05e-665578e1bba1
countours = find_contours(Gray.(response.>quantile(vec(response),.9)))

# ╔═╡ 7e182d8b-aeed-486b-9873-50ece1b8a219
response_cp = copy(response)

# ╔═╡ c03c102e-cade-435a-90ef-ffabc2a97c7d
draw_contours(response_cp,RGB(1,0,0),countours)

# ╔═╡ 17837126-5c77-445b-a303-6adcff5df88c
response_cp

# ╔═╡ 967046cf-90ce-4a5f-86f5-9fcc7af4b86b
function ROI(img,center,radius)
	x,y = Tuple(center)
	padding = Integer(round(radius))
	xs,ys = size(img)
	img[max(x-padding,1):min(x+padding,xs),max(y-padding,1):min(y+padding,ys)]
end

# ╔═╡ 56862b73-4a41-4b1d-9370-1aca95380f35
irisROI = ROI(sample,center,haarRadius)

# ╔═╡ 856b8c4f-88e0-4f23-b015-997ed6c8c6c6
begin
	plot([histogram(vec(channelview(irisROI)[i,:,:])) for i in 1:3]...
	,layout=(3,1),legend=false)
end

# ╔═╡ c3981677-bf5a-45d1-8680-e023ba1a9093
histogram(vec(channelview(irisROI)[1,:,:]))

# ╔═╡ 5b98504c-5f98-499d-ac2b-f165eacedb7e
irisThreshold = [quantile(vec(channelview(irisROI)[i,:,:]),PercentageInliers) for i in 1:3]

# ╔═╡ f8fa6ecc-26fb-4188-a212-859903c76361
quantile(vec(response),.85)

# ╔═╡ b481e342-0651-4dcf-8a9d-41a4ed3ca9dd
argmin(response)

# ╔═╡ 24419aba-ad2a-4681-b3c5-3fa5fe1c35e4
response[argmax(response)]

# ╔═╡ 141688e5-dc5e-4b65-a97a-e2752eedcf5b
seeds = [(center,1),(argmin(response),2)]

# ╔═╡ ee695bfd-38af-44c9-80e3-84d13a7e8b9f
begin
	irisMask = Gray.(cmp(float64.(sample), irisThreshold,<)) .>= 1
	Gray.(irisMask)
end

# ╔═╡ cf512220-3d3d-40d0-af17-824f310e93b4
sample .* (Gray.(cmp(float64.(sample), irisThreshold,<)) .>= 1)

# ╔═╡ 179f5e20-f4eb-4f05-af43-c56cbef25060
gaussianKernel = ImageFiltering.Kernel.gaussian((1,1),(3,3))

# ╔═╡ b776e384-0777-4786-9c62-c940e96a1780
(dx,dy) = ImageFiltering.Kernel.sobel().*1

# ╔═╡ fed69afa-d247-408e-9360-2e0709d90e6c
Laplacian = ImageFiltering.Kernel.Laplacian()


# ╔═╡ ab048149-e900-4355-9c04-a016d8000d94
kerenels = [gaussianKernel,Laplacian]

# ╔═╡ f322dc9d-dbf1-4e3c-908a-a4f829c1fb18
begin
	irisOutline = reduce((u,v)->imfilter(u,v),kerenels;init=irisMask) .> 0.
	Gray.(irisOutline)
end

# ╔═╡ a3fda49b-1b62-4e10-b0e9-3cabdd744aab
find_contours(irisOutline)

# ╔═╡ 999dd8e7-d4b8-4dcd-93e6-46984d3b54d9
canny(δ) = Canny(spatial_scale=δ, low=Percentile(CannyMinThreshold), high=Percentile(CannyMaxThreshold))

# ╔═╡ ebe4d314-13b0-4c9e-a760-9fc15e757b41
contours = find_contours(irisOutline)

# ╔═╡ 05748b0a-3fa4-4e1e-8344-26fc09d15618
points =collect.(Tuple.(vcat(contours...)))

# ╔═╡ 42358388-f973-4f90-b601-e547ade6afd2
function fitEclipse(points)
	#initial Value
	N = size(points)
	x0,y0 = mean(points)
	X = map(x->x[1],points)
	Y = map(x->x[2],points)
	A = [X.^2 Y.^2 X.*Y X Y ones(N)]

	p0 = A\ones(N)
	@. eclipse_poly(p) = p[1]*X^2 + p[2]*Y^2 + p[3]*X*Y + p[4]*X + p[5]*Y + p[6]
	J(x,y) = a
	
end

# ╔═╡ 582ecf2a-a877-48b7-9224-28f42bdd6f38
function cononical_form(p)
	A,B,C,D,E,F = p
	D = (A-C)^2+B^2
	a,b = -(2*(A*E^2+C*D^2-B*D*E+(B^2-4*A*C)*F)*((A+C)))^0.5/(B^2-4*A*C)
end

# ╔═╡ 25f01b2a-4fb8-4f0a-aefd-d7a501ba514c
begin
	X = map(x->x[1],points)
	Y = map(x->x[2],points)
	A = [X.^2 Y.^2 X.*Y X Y ones(size(points))]
	A\ones(size(points))
end

# ╔═╡ 18a3dd52-18d5-4ae0-aaf5-7541a687d152
eigen(A'*A)

# ╔═╡ Cell order:
# ╠═cd9800b0-bb3a-11ec-1091-7985413e1e8e
# ╠═42635499-73a1-4a2b-81be-b73cacb8db7c
# ╠═fdfd2b19-a817-4a9d-910e-3a17229f7907
# ╠═0ed7820d-ec75-4a47-9679-e69cc2c7cd17
# ╠═325d5ac4-e710-4881-8279-6c569ba76509
# ╠═771124c2-7e8a-41f9-bfbf-987121481910
# ╠═f1b5f625-487c-4955-a37c-afab56b03afd
# ╠═2c14fc4d-a3cb-4bc1-a802-918c8fb11f3c
# ╠═f9e95fa5-8120-4b6c-bd29-9c91a5fc64d4
# ╠═fa6eed51-f831-4d0a-8ac2-78211170cf80
# ╠═f05bc3ce-634c-4dca-b8d5-8c6cc0b7e860
# ╠═96db2505-4134-4d23-96cc-81ebb2247564
# ╠═8200374b-4ef4-452b-bf8b-862775bbd00e
# ╠═51edc0ff-85e2-441b-9ad8-9f3902c3250f
# ╠═83bce8f4-459b-400c-9be7-fa985169acf3
# ╠═537ec3f4-4f90-41f1-88df-41517b925ec2
# ╠═63b77666-ea9a-40a8-a66a-fd5339cd8ce5
# ╠═60955ee9-3e37-49bb-92ba-0b2dc7c7d1c2
# ╠═13db6529-c736-4524-9396-b3b461a380fe
# ╠═f4fdd861-11af-447c-bb02-0d4f76dbb4a6
# ╠═93dba41e-4eb7-436e-93d2-b4d4be23dfdb
# ╠═bcd7d54d-7506-48e5-8e1b-08955143c8ec
# ╠═8aaeae0e-1e11-474c-8926-178840367d16
# ╠═c523fcae-6627-493a-951b-7b8e585fe52f
# ╠═598c5201-2409-43fd-8673-5a827db79b04
# ╠═2955fbde-5718-4ed1-ade5-e2466318bdf4
# ╠═8b029d5b-becf-4de7-a05e-665578e1bba1
# ╠═7e182d8b-aeed-486b-9873-50ece1b8a219
# ╠═c03c102e-cade-435a-90ef-ffabc2a97c7d
# ╠═17837126-5c77-445b-a303-6adcff5df88c
# ╠═967046cf-90ce-4a5f-86f5-9fcc7af4b86b
# ╠═56862b73-4a41-4b1d-9370-1aca95380f35
# ╠═856b8c4f-88e0-4f23-b015-997ed6c8c6c6
# ╠═c3981677-bf5a-45d1-8680-e023ba1a9093
# ╠═5b98504c-5f98-499d-ac2b-f165eacedb7e
# ╠═f8fa6ecc-26fb-4188-a212-859903c76361
# ╠═b481e342-0651-4dcf-8a9d-41a4ed3ca9dd
# ╠═24419aba-ad2a-4681-b3c5-3fa5fe1c35e4
# ╠═141688e5-dc5e-4b65-a97a-e2752eedcf5b
# ╠═ee695bfd-38af-44c9-80e3-84d13a7e8b9f
# ╠═cf512220-3d3d-40d0-af17-824f310e93b4
# ╠═179f5e20-f4eb-4f05-af43-c56cbef25060
# ╠═b776e384-0777-4786-9c62-c940e96a1780
# ╠═fed69afa-d247-408e-9360-2e0709d90e6c
# ╠═ab048149-e900-4355-9c04-a016d8000d94
# ╠═f322dc9d-dbf1-4e3c-908a-a4f829c1fb18
# ╠═a3fda49b-1b62-4e10-b0e9-3cabdd744aab
# ╠═999dd8e7-d4b8-4dcd-93e6-46984d3b54d9
# ╠═ebe4d314-13b0-4c9e-a760-9fc15e757b41
# ╠═05748b0a-3fa4-4e1e-8344-26fc09d15618
# ╠═42358388-f973-4f90-b601-e547ade6afd2
# ╠═582ecf2a-a877-48b7-9224-28f42bdd6f38
# ╠═25f01b2a-4fb8-4f0a-aefd-d7a501ba514c
# ╠═18a3dd52-18d5-4ae0-aaf5-7541a687d152
