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
	,"Rotations","ProgressMeter","PlutoUI","LaTeXStrings","HypertextLiteral","Plots","PlotlyJS","DataFrames","ImageSegmentation","ImageEdgeDetection","LsqFit","JuMP","Markdown", "InteractiveUtils"])
	using Images, ImageCore
	using Plots, DataFrames
	using ImageSegmentation, ImageEdgeDetection
	using ImageEdgeDetection: Percentile
	using LinearAlgebra,JuMP
	using Statistics
	using LsqFit
	using ImageIO, VideoIO
	using ProgressMeter,PlutoUI,LaTeXStrings
	using ImageFeatures, ImageDraw, CoordinateTransformations, Rotations
	using Markdown, InteractiveUtils,HypertextLiteral

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
	PercentageInliers = .4
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

# ╔═╡ fcc9384f-3ce3-421a-94c6-c1cbd4af7f91
hsv_irisROI = HSV.(irisROI)

# ╔═╡ 856b8c4f-88e0-4f23-b015-997ed6c8c6c6
begin
	plot([histogram(vec(channelview(irisROI)[i,:,:])) for i in 1:3]...
	,layout=(3,1),legend=false)
end

# ╔═╡ b9e3c0c0-afe8-4870-9cc5-7098a03e93f0
begin
	plot([histogram(vec(channelview(hsv_irisROI)[i,:,:])) for i in 1:3]...
	,layout=(3,1),legend=false)
end

# ╔═╡ c907ebe9-d0d5-4fd6-aefa-e0e6a4002034
H,S,V = [channelview(hsv_irisROI)[i,:,:] for i in 1:3]

# ╔═╡ 801ddea6-8b89-47f0-8a7c-a8f00a03fbcb
mH,mS,mV = [channelview(sample)[i,:,:] for i in 1:3]

# ╔═╡ 1e05e95c-ea34-4327-9698-37195db35824
irisROI .* (H .<= quantile(vec(H), .7) .&& H .>= quantile(vec(H), .3))

# ╔═╡ e29d9583-37c8-4088-874f-ca12fba2ff26
sample .* (mV .<= quantile(vec(V), .3))

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

# ╔═╡ bf2603a3-fdf1-4a75-8d4c-fc926e3623db
points =collect.(Tuple.(vcat(contours...)))

# ╔═╡ 58271ca0-4731-46aa-b848-6d365f0a1e48
md"""
# Ellipse Fitting 
"""

# ╔═╡ a3650f78-d7fe-4da9-bbaa-7d411c7672f0
md"""
## Problem Statement
```math
F(A,X) = A \cdot X = ax^2 + bxy + cy^2 + dx + ey + f = 0 
```
$F(A,X) \ \text{is the distance function for the Ellipse to data points}$ 
"""

# ╔═╡ 74dd4f1c-e285-440c-95eb-831591a9b5ea
L"A = [a\  b\  c\ d\ e\ f]^T"

# ╔═╡ 6b052ca4-8296-4d52-86d3-0026b97468ce
L"X = [x^2\ xy\ y^2 x\ y\ ]^T"

# ╔═╡ 63cd2bea-97dc-4c79-996a-294afe9f58e5
md"""
## Contstrain 

By forcing the discriminant $b^2 - 4ac \leq 0$,

"""

# ╔═╡ 53667ef2-2b4f-4981-96f2-902dddef55fd
L"A^T\begin{bmatrix}
0 & 0 & 2 & 0 & 0 & 0\\
0 & -1 & 0 & 0 & 0 & 0\\
2 & 0 & 0 & 0 & 0 & 0\\
0 & 0 & 0 & 0 & 0 & 0\\
0 & 0 & 0 & 0 & 0 & 0\\
0 & 0 & 0 & 0 & 0 & 0\\
\end{bmatrix}A=1"

# ╔═╡ 482649ae-6747-4f29-852d-bb32debb9f3c
C = [ 	0 0 2 0 0 0
		0 -1 0 0 0 0 
		2 0 0 0 0 0 
		0 0 0 0 0 0 
		0 0 0 0 0 0 
		0 0 0 0 0 0 ]

# ╔═╡ 3ec810d2-ba8e-4ac0-81e7-a10ddbcd74a5
md"""
then  the contstrain fitting problem

$\text{minimize }  J = \begin{Vmatrix}XA\end{Vmatrix}^2$
"""

# ╔═╡ f197e74c-6de9-4301-b0e8-f1357d5ea2cc
# The Cost Function
J(X,A) = norm(X*A,2)

# ╔═╡ 3c67f7fe-0c09-48ef-a268-c80f0d2629bd
md"""
we obtained the system of equations
```math
\begin{align}
            X^TXA &= \lambda CA\\ 
            A^TCA &= 1\\
\end{align}

```


"""

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

# ╔═╡ 25f01b2a-4fb8-4f0a-aefd-d7a501ba514c
begin
	x = map(x->x[1],points)
	y = map(x->x[2],points)
	X = [x.^2 y.^2 x.*y x y ones(size(points))]
	X\ones(size(points))
end

# ╔═╡ 9e39afa9-efff-4d6b-9917-e005ce12e393
D = reduce(hcat,points)

# ╔═╡ 51311909-9f3c-44e0-a6ca-b92534ec18a9
D'D

# ╔═╡ d1ce0371-98c5-41a1-8e9c-630fd0075a27
md"""
```math
\mu_i = \sqrt{\frac{1}{u_i^TCu_i}} = \sqrt{\frac{\lambda_i}{u_i^TSu_i}}
```
##### There should be a minimum positive defined eigenvalue
"""

# ╔═╡ 18a3dd52-18d5-4ae0-aaf5-7541a687d152
genval , genvec = eigen(X'*X,C)

# ╔═╡ 80de1f18-7494-4296-af0c-3b2c24bbc45d
md"""
###### TODO:
Final Eigen Value and Eigne Vector, Foreach Eigen Value and Vector obtain the scaling factor
Then find the minimum noise (eigenval,eigenvector) pair
"""

# ╔═╡ 44ed9c0c-1000-48db-b70f-2983ccf080a9
function minimizePair(X,λ,v;C=C) # -> minimum noise pair of (λ,v) with sacling factor μ
	S = X'X
	c1(S,λ,v) =  S*v - λ.*C*v
	c2(v,C) = v'*C*v - 1
	μ(S,λ,v) = sqrt(λ/(v'*S*v))
	posIdx = (λ .!= Inf .&& λ .> 0)
	λ,v = λ[posIdx],v[:,posIdx]
	if(length(λ) == 1)
		return (λ,μ(S,λ,v).*v)
	else
		idx = argmin([ norm((μ(S,λ[i],v[:,i]) .* v[:,i])' * S * (μ(S,λ[i],v[:,i]) .* v[:,i]) - λ[i]) for i in 1:length(λ)])
		return (λ(idx),v[:,idx])
	end
	
end

# ╔═╡ f72e821a-08ff-4b7b-950d-68f914fa193b
function directLsqSolve(X;C=C)
	genval , genvec = eigen(X'*X\C)
	(λ,v)=minimizePair(X,genval,genvec)
	a = λ.*v
	return (λ,v)
end

# ╔═╡ 8168834e-072b-4d23-b0f5-540e34c2fe6c
function fitEllipse(points)
	x = map(x->x[1],points)
	y = map(x->x[2],points)
	X = [x.^2 y.^2 x.*y x y ones(size(points))]
	a = directLsqSolve(X)
	
	
end

# ╔═╡ 755442fe-22c3-41c2-a5d5-238fdee7a21d
λ,v = fitEllipse(points)

# ╔═╡ 92f3be15-484e-4fcc-828b-4b9a15042f2b
â = λ .* v

# ╔═╡ 89170295-4ff9-4de3-b96c-f70622e11e69
norm(X*â)

# ╔═╡ 8db21d64-643b-44aa-a042-f9af71677cdf
±(x, y) = [x+y, x-y]

# ╔═╡ f03d0e7c-71d6-45ff-a387-aac0c66d507b
function cononical_form(p)
	A,B,C,D,E,F = p
	Δ = B^2 - 4*A*C
	δ = (A-C)^2+B^2
	a,b = -(2*(A*E^2+C*D^2-B*D*E+(B^2-4*A*C)*F)*((A+C)±δ)).^0.5./(B^2-4*A*C)
	xₒ = (2*C*D-B*E)/(Δ)
	yₒ = (2*A*E - B*D)/(Δ)
	θ = if(B==0 && A<C)
		0
	elseif(B==0 && A>C)
		π/2
	else
		atan((1/B)*(C-A-sqrt((A-C)^2+B^2)))
	end
	return (a,b,xₒ,yₒ,θ)
end

# ╔═╡ 752d47d9-f3f5-4620-a1f2-b95d030decda
#p,q,xₒ,yₒ,θ = cononical_form(â)

# ╔═╡ 3913502b-1f49-4033-85da-9575735fc5bb
p,q,xₒ,yₒ,θ = cononical_form(v)

# ╔═╡ b707cdf0-c7db-4448-8088-d68b6335b17d
sample_copy = copy(sample)

# ╔═╡ cf516d74-644f-44d7-9725-5662b7ef6fdb
draw(sample_copy,Cross(Point(round(xₒ),round(yₒ)), 50), RGB{N0f8}(0,1,0))

# ╔═╡ 17c3396c-e74d-4472-841b-e935f021e009
draw(sample_copy,Ellipse(Point(round(xₒ),round(yₒ)),q,p; thickness=75,fill = false), RGB{N0f8}(1,0,0))

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
# ╠═fcc9384f-3ce3-421a-94c6-c1cbd4af7f91
# ╠═856b8c4f-88e0-4f23-b015-997ed6c8c6c6
# ╠═b9e3c0c0-afe8-4870-9cc5-7098a03e93f0
# ╠═c907ebe9-d0d5-4fd6-aefa-e0e6a4002034
# ╠═801ddea6-8b89-47f0-8a7c-a8f00a03fbcb
# ╠═1e05e95c-ea34-4327-9698-37195db35824
# ╠═e29d9583-37c8-4088-874f-ca12fba2ff26
# ╠═c3981677-bf5a-45d1-8680-e023ba1a9093
# ╠═5b98504c-5f98-499d-ac2b-f165eacedb7e
# ╠═f8fa6ecc-26fb-4188-a212-859903c76361
# ╠═b481e342-0651-4dcf-8a9d-41a4ed3ca9dd
# ╠═24419aba-ad2a-4681-b3c5-3fa5fe1c35e4
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
# ╠═bf2603a3-fdf1-4a75-8d4c-fc926e3623db
# ╠═58271ca0-4731-46aa-b848-6d365f0a1e48
# ╠═a3650f78-d7fe-4da9-bbaa-7d411c7672f0
# ╠═74dd4f1c-e285-440c-95eb-831591a9b5ea
# ╠═6b052ca4-8296-4d52-86d3-0026b97468ce
# ╠═63cd2bea-97dc-4c79-996a-294afe9f58e5
# ╠═53667ef2-2b4f-4981-96f2-902dddef55fd
# ╠═482649ae-6747-4f29-852d-bb32debb9f3c
# ╠═3ec810d2-ba8e-4ac0-81e7-a10ddbcd74a5
# ╠═f197e74c-6de9-4301-b0e8-f1357d5ea2cc
# ╠═3c67f7fe-0c09-48ef-a268-c80f0d2629bd
# ╠═42358388-f973-4f90-b601-e547ade6afd2
# ╠═25f01b2a-4fb8-4f0a-aefd-d7a501ba514c
# ╠═89170295-4ff9-4de3-b96c-f70622e11e69
# ╠═9e39afa9-efff-4d6b-9917-e005ce12e393
# ╠═51311909-9f3c-44e0-a6ca-b92534ec18a9
# ╠═d1ce0371-98c5-41a1-8e9c-630fd0075a27
# ╠═18a3dd52-18d5-4ae0-aaf5-7541a687d152
# ╠═80de1f18-7494-4296-af0c-3b2c24bbc45d
# ╠═44ed9c0c-1000-48db-b70f-2983ccf080a9
# ╠═f72e821a-08ff-4b7b-950d-68f914fa193b
# ╠═8168834e-072b-4d23-b0f5-540e34c2fe6c
# ╠═755442fe-22c3-41c2-a5d5-238fdee7a21d
# ╠═92f3be15-484e-4fcc-828b-4b9a15042f2b
# ╠═8db21d64-643b-44aa-a042-f9af71677cdf
# ╠═f03d0e7c-71d6-45ff-a387-aac0c66d507b
# ╠═752d47d9-f3f5-4620-a1f2-b95d030decda
# ╠═3913502b-1f49-4033-85da-9575735fc5bb
# ╠═b707cdf0-c7db-4448-8088-d68b6335b17d
# ╠═cf516d74-644f-44d7-9725-5662b7ef6fdb
# ╠═17c3396c-e74d-4472-841b-e935f021e009
