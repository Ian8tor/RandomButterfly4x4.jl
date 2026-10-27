module RandomButterfly4x4
    
	#=============== Imports =============================#
	#using JLD2
	#using Plots
	using Random
	#using Profile
	#using StatsPlots
	#using PkgTemplates
	using Distributions
	using LinearAlgebra
	#using BenchmarkTools



	#=============== Growth Factor Computation ===========#
	function findGrowthInf!(A, pivot_strag = RowMaximum())
		try
			a = opnorm(A, Inf)
			lu!(A, pivot_strag)

			l = opnorm(UnitLowerTriangular(A), Inf)
			u = opnorm(UpperTriangular(A)    , Inf)
			return ((l * u) / a)
		catch
			return -1
		end
	end

	function findGrowthInf(A, pivot_strag = RowMaximum())
		try
			ludecomp = lu(A, pivot_strag)

			a = opnorm(A, Inf) 
			l = opnorm(ludecomp.L, Inf)
			u = opnorm(ludecomp.U, Inf)
			return ((l * u) / a)
		catch
			return -1
		end
	end



	#=============== Creates Matrixes ====================#
	function findBadMatrix(iterations, size, pivot_strat = NoPivot(), dist = Uniform(-1, 1))
		worst = rand(dist, (size, size))
		gf = findGrowthInf(worst, pivot_strat)
		mat = worst
		
		for i in 1:(iterations-1)
			mat = rand(dist, (size, size))
			g = findGrowthInf(mat, pivot_strat)
			if (gf < g)
				gf = g
				worst = mat
			end
		end
		return worst
	end

	function createPatternBad(size, e)
		mat = Matrix(0.0I, size, size)
		mat[1, 1] = e
		for i in 1:(size - 1)
			mat[i  , i+1] = 1
			mat[i+1, i  ] = 1
		end

		return mat
	end

	function wilkinsonMat(size)
		mat = Matrix(1.0I, size, size)
		for y in 1:size
			mat[y, size] = 1
			for x in 1:(y-1)
				mat[y, x] = -1
			end
		end
		return mat
	end



	#=============== Creates a butterfly matrix ==========# 
	function createGivens(ps)
		if size(ps)[1] == 2
			return [ps[1] ps[2]; 
				   -ps[2] ps[1]]
		end
		if size(ps)[1] == 4
			return [ ps[1]  ps[2]  ps[3]  ps[4]; 
					-ps[2]  ps[1]  ps[4] -ps[3]; 
					-ps[3] -ps[4]  ps[1]  ps[2]; 
					-ps[4]  ps[3] -ps[2]  ps[1]]
		end
		if size(ps)[1] == 8
			return [ ps[1]  ps[2]  ps[3]  ps[4]  ps[5]  ps[6]  ps[7]  ps[8];
					-ps[2]  ps[1]  ps[4] -ps[3]  ps[6] -ps[5] -ps[8]  ps[7];
					-ps[3] -ps[4]  ps[1]  ps[2]  ps[7]  ps[8] -ps[5] -ps[6];
					-ps[4]  ps[3] -ps[2]  ps[1]  ps[8] -ps[7]  ps[6] -ps[5];
					-ps[5] -ps[6] -ps[7] -ps[8]  ps[1]  ps[2]  ps[3]  ps[4];
					-ps[6]  ps[5] -ps[8]  ps[7] -ps[2]  ps[1] -ps[4]  ps[3];
					-ps[7]  ps[8]  ps[5] -ps[6] -ps[3]  ps[4]  ps[1] -ps[2];
					-ps[8] -ps[7]  ps[6]  ps[5] -ps[4] -ps[3]  ps[2]  ps[1]]
		end

	   


	end

	function createButterfly(size, depth, givens = 2, simple=true, seed=rand(Int64))
		rng = MersenneTwister(seed)
		dim = givens^(size-1)

		Ps = Array{Any}(undef, givens)
		for i in 1:givens
		   Ps[i] = Matrix(1.0I, dim, dim)
		end

		#Creates the Cs and Ns such that them squared add to identity#
		for i in 1:dim
			vals = rand(rng, Uniform(-1, 1), (givens))
			vals /= sqrt(dot(vals, vals))
			for j in 1:givens
				Ps[j][i, i] = vals[j]
			end
		end
		butterfly = createGivens(Ps)
		
		if (depth == 1)
			return butterfly
		end

		Z = Matrix(0I, dim, dim)
		As = Array{Any}(undef, givens)
		As[1] = createButterfly(size-1, depth-1, givens, simple, rand(rng, Int64))
		for i in 2:givens
			if simple
				As[i] = As[1]
			else
				As[i] = createButterfly(size-1, depth-1, givens, simple, rand(rng, Int64))
			end
		end

		if givens == 2
			return butterfly * [As[1] Z; Z As[2]]
		end
		if givens == 4
			return butterfly * [As[1] Z Z Z; Z As[2] Z Z; Z Z As[3] Z; Z Z Z As[4]]
		end
	end



	#=============== Returns depths of the butterfly =====#
	function getButterflies(matSize, givens = 2, seed=rand(Int64))
		rng = MersenneTwister(seed)

		#Used to get the butterflies at each layer
		Ps = Array{Any}(undef, givens)
		butterflies = Array{Any}(undef, matSize)
		
		for depth in 1:matSize
			dim = givens^(matSize-depth)

			#Creates the diagonal P matrixes
			for mat in 1:givens
				Ps[mat] = Matrix(1.0I, dim, dim)
				for i in 1:dim
					vals = rand(rng, Uniform(-1, 1), (givens))
					vals /= sqrt(dot(vals, vals))
					for j in 1:givens
						Ps[mat][i, i] = vals[j]
					end
				end
			end

			#Uses the givens function to make a givens matrix with the P matrixes#
			B = createGivens(Ps)
			for i in 1:(depth-1)
				Z = Matrix(0.0*I, size(B)[1], size(B)[2])
				if (givens == 2) B = [B Z; Z B] end
				if (givens == 4) B = [B Z Z Z; Z B Z Z; Z Z B Z; Z Z Z B] end
				if (givens == 8) B = [B Z Z Z Z Z Z Z; Z B Z Z Z Z Z Z; Z Z B Z Z Z Z Z; Z Z Z B Z Z Z Z; Z Z Z Z B Z Z Z; Z Z Z Z Z B Z Z; Z Z Z Z Z Z B Z; Z Z Z Z Z Z Z B] end
			end
			
			#Adds the butterfly to the list#
			butterflies[depth] = B
			
		end

		return butterflies
	end



	#=============== Gets data ===========================#
	function getData(A, trials, givens, dim)
		#Creates the matrix to hold the data#
		max_depth = Int64(log2(dim)/log2(givens))
		dataTopDown  = Matrix{Float64}(undef, trials, max_depth)
		dataDownTop  = Matrix{Float64}(undef, trials, max_depth)

		for trial in 1:trials
			#Gets the set of random butterflies#
			butterflies = getButterflies(max_depth, givens, trial)
			topButterfly = Matrix(1.0I, size(butterflies[1  ])[1], size(butterflies[1  ])[2])
			botButterfly = Matrix(1.0I, size(butterflies[end])[1], size(butterflies[end])[2])
		
			for depth in 1:max_depth
				#Creates the different depth butterflies#
				topButterfly *= butterflies[depth]
				botButterfly *= butterflies[end - (depth-1)]

				#Adds the data#
				dataTopDown[trial, depth] = findGrowthInf!(topButterfly * bad, NoPivot())
				dataDownTop[trial, depth] = findGrowthInf!(botButterfly * bad, NoPivot())
			end
		end

		#Returns the data#
		return (dataTopDown, dataDownTop)
	end


end
