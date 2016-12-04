using ArrayFire
setBackend(AF_BACKEND_OPENCL)

const GPU_SUPPORTS_FLOAT64 = false
if GPU_SUPPORTS_FLOAT64
	const gpu_float = Float64
else
	const gpu_float = Float32
end

function rand_arr(a::Float64,b::Float64,dims::Int...)
    arr = zeros(AFArray{gpu_float},dims)
    rand!(arr)
    arr*(Float32(b)-Float32(a))+Float32(a)
end

function sigmoid_helper(x::gpu_float)
    return 1/(1+exp(-x))
end

function sigmoid(x::Union{AFArray{gpu_float,1},gpu_float})
    try
        return sigmoid_helper(x)
    catch
        return map(sigmoid_helper,x)
    end
end

function mat_dot(a::Union{AFArray{gpu_float,1},AFArray{gpu_float,2}},b::AFArray{gpu_float,1})
    try
        return dot(a,b)
    catch
        *(a,b)
    end
end

function outer(a::AFArray{gpu_float,1},b::AFArray{gpu_float,1})
    *(vec(a),transpose(vec(b)))
end

function dot_transpose(a::Union{AFArray{gpu_float,1},AFArray{gpu_float,2}},b::AFArray{gpu_float,1})
    mat_dot(transpose(a),b)
end

function LSTM_Param(mem_cell_ct::Int,x_dim::Int)
    concat_len = x_dim+mem_cell_ct
    Dict("wg"=> rand_arr(-0.1,0.1,mem_cell_ct,concat_len),
         "wi"=> rand_arr(-0.1,0.1,mem_cell_ct,concat_len),
         "wf"=> rand_arr(-0.1,0.1,mem_cell_ct,concat_len),
         "wo"=> rand_arr(-0.1,0.1,mem_cell_ct,concat_len),
         "bg"=> rand_arr(-0.1,0.1,mem_cell_ct),
         "bi"=> rand_arr(-0.1,0.1,mem_cell_ct),
         "bf"=> rand_arr(-0.1,0.1,mem_cell_ct),
         "bo"=> rand_arr(-0.1,0.1,mem_cell_ct), 
         "wg_diff"=> zeros(AFArray{gpu_float},(mem_cell_ct, concat_len)),
         "wi_diff"=> zeros(AFArray{gpu_float},(mem_cell_ct, concat_len)),
         "wf_diff"=> zeros(AFArray{gpu_float},(mem_cell_ct, concat_len)),
         "wo_diff"=> zeros(AFArray{gpu_float},(mem_cell_ct, concat_len)),
         "bg_diff"=> zeros(AFArray{gpu_float},mem_cell_ct),
         "bi_diff"=> zeros(AFArray{gpu_float},mem_cell_ct),
         "bf_diff"=> zeros(AFArray{gpu_float},mem_cell_ct),
         "bo_diff"=> zeros(AFArray{gpu_float},mem_cell_ct),
         "concat_len"=> concat_len,
         "mem_cell_ct"=> mem_cell_ct,
         "x_dim"=> x_dim
    )
end

function apply_diff(params_dict::Dict{String,Any}, lr::gpu_float)
    params = params_dict
    
    params["wg"] -= lr*params["wg_diff"]
    params["wi"] -= lr*params["wi_diff"]
    params["wf"] -= lr*params["wf_diff"]
    params["wo"] -= lr*params["wo_diff"]
    params["bg"] -= lr*params["bg_diff"]
    params["bi"] -= lr*params["bi_diff"]
    params["bf"] -= lr*params["bf_diff"]
    params["bo"] -= lr*params["bo_diff"]

    params["wg_diff"] = zeros(AFArray{gpu_float},size(params["wg"]))
    params["wi_diff"] = zeros(AFArray{gpu_float},size(params["wi"])) 
    params["wf_diff"] = zeros(AFArray{gpu_float},size(params["wf"])) 
    params["wo_diff"] = zeros(AFArray{gpu_float},size(params["wo"])) 
    params["bg_diff"] = zeros(AFArray{gpu_float},size(params["bg"]))
    params["bi_diff"] = zeros(AFArray{gpu_float},size(params["bi"])) 
    params["bf_diff"] = zeros(AFArray{gpu_float},size(params["bf"])) 
    params["bo_diff"] = zeros(AFArray{gpu_float},size(params["bo"])) 
    params
end

abstract LSTM

function LSTM_state(mem_cell_ct::Int, x_dim::Int)
    Dict("g"=> zeros(AFArray{gpu_float},mem_cell_ct),
         "i"=> zeros(AFArray{gpu_float},mem_cell_ct),
         "f"=> zeros(AFArray{gpu_float},mem_cell_ct),
         "o"=> zeros(AFArray{gpu_float},mem_cell_ct),
         "s"=> zeros(AFArray{gpu_float},mem_cell_ct),
         "h"=> zeros(AFArray{gpu_float},mem_cell_ct),
         "bottom_diff_h"=> zeros(AFArray{gpu_float},mem_cell_ct),
         "bottom_diff_s"=> zeros(AFArray{gpu_float},mem_cell_ct),
         "bottom_diff_x"=> zeros(AFArray{gpu_float},x_dim),
         "mem_cell_ct"=> mem_cell_ct,
         "x_dim"=> x_dim)
end

type LSTM_Node <: LSTM
    param::Dict{String,Any}
    state::Dict{String,Any}
    x::Union{AFArray{gpu_float,1},Void}
    xc::Union{AFArray{gpu_float,1},Void}
    s_prev::Union{AFArray{gpu_float,1},Void}
    h_prev::Union{AFArray{gpu_float,1},Void}
end

function bottom_data_is(self::LSTM_Node,x::AFArray{gpu_float,1},s_prev=nothing::Union{Void,AFArray},h_prev=nothing::Union{Void,AFArray})
    if s_prev==nothing
        s_prev = zeros(AFArray{gpu_float},self.state["s"])
    end
    if h_prev==nothing
        h_prev = zeros(AFArray{gpu_float},self.state["h"])
    end
    self.s_prev=AFArray(vec(s_prev))
    self.h_prev=AFArray(vec(h_prev))
    
    xc = AFArray(vec(vcat(vec(x), vec(h_prev))))
        
    self.state["g"] = tanh(mat_dot(self.param["wg"],xc)+self.param["bg"])
    self.state["i"] = sigmoid(mat_dot(self.param["wi"],xc)+self.param["bi"])
    self.state["f"] = sigmoid(mat_dot(self.param["wf"],xc)+self.param["bf"])
    self.state["o"] = sigmoid(mat_dot(self.param["wo"],xc)+self.param["bo"])
    self.state["s"] = self.state["g"].*self.state["i"]+s_prev.*self.state["f"]
    self.state["h"] = self.state["s"].*self.state["o"]
    self.x = AFArray(vec(x))
    self.xc = AFArray(vec(xc))
    
    LSTM_Node
end

function top_diff_is(self::LSTM_Node,top_diff_h::AFArray{gpu_float,1},top_diff_s::AFArray{gpu_float,1})
    ds  = self.state["o"].*top_diff_h+top_diff_s
    do_ = self.state["s"].*top_diff_h # underscore because do is reserved
    di  = self.state["g"].*ds
    dg  = self.state["i"].*ds
    df  = self.s_prev.*ds
    
    di_input = (1-self.state["i"]).*self.state["i"].*di
    df_input = (1-self.state["f"]).*self.state["f"].*df
    do_input = (1-self.state["o"]).*self.state["o"].*do_
    dg_input = (1-(self.state["g"].*self.state["g"])).*dg
    
    self.param["wi_diff"] += outer(di_input,self.xc)
    self.param["wf_diff"] += outer(df_input,self.xc)
    self.param["wo_diff"] += outer(do_input,self.xc)
    self.param["wg_diff"] += outer(dg_input,self.xc)
    self.param["bi_diff"] += di_input
    self.param["bf_diff"] += df_input
    self.param["bo_diff"] += do_input
    self.param["bg_diff"] += dg_input
    
    dxc = zeros(AFArray{gpu_float},size(self.xc))
    dxc += dot_transpose(self.param["wi"], di_input)
    dxc += dot_transpose(self.param["wf"], df_input)
    dxc += dot_transpose(self.param["wo"], do_input)
    dxc += dot_transpose(self.param["wg"], dg_input)
    
    self.state["bottom_diff_s"] = ds.*self.state["f"]
    self.state["bottom_diff_x"] = dxc[1:self.param["x_dim"]+1]
    self.state["bottom_diff_h"] = dxc[self.param["x_dim"]+1:end]
end

type LossLayer
end

function loss_func(self::Type{LossLayer}, pred::AFArray{gpu_float,1}, label::gpu_float)
    (pred[1]-label)^2 #squared error
end

function bottom_diff(self::Type{LossLayer}, pred::AFArray{gpu_float,1}, label::gpu_float)
    diff = zeros(AFArray{gpu_float},pred)
    diff[1] = 2*(pred[1]-label)
    diff
end

type LSTM_Network <: LSTM
    lstm_param::Dict{String,Any}
    lstm_node_list::Array{LSTM_Node,1}
    x_list
end


function y_list_is(self::LSTM_Network,y_list::AFArray{gpu_float,1},loss_layer::Type{LossLayer})
    if length(y_list) != length(self.x_list)
        error("AssertionError: length(y_list) is not equal to length(self.x_list)")
    end
    idx = length(self.x_list)-1
    loss = loss_func(loss_layer, self.lstm_node_list[idx+1].state["h"], y_list[idx+1])
    diff_h = bottom_diff(loss_layer, self.lstm_node_list[idx+1].state["h"], y_list[idx+1])
    diff_s = zeros(AFArray{gpu_float},self.lstm_param["mem_cell_ct"])
    top_diff_is(self.lstm_node_list[idx+1],diff_h,diff_s)
    idx -= 1
    while idx >= 0
        loss += loss_func(loss_layer, self.lstm_node_list[idx+1].state["h"], y_list[idx+1])
        diff_h = bottom_diff(loss_layer, self.lstm_node_list[idx+1].state["h"], y_list[idx+1])
        diff_h += self.lstm_node_list[idx+1].state["bottom_diff_h"]
        diff_s = self.lstm_node_list[idx+1].state["bottom_diff_s"]
        top_diff_is(self.lstm_node_list[idx+1],diff_h,diff_s)
        idx -= 1
    end
    loss
end

function x_list_clear(self::LSTM_Network)
	self.x_list=Array{Any}(Array{gpu_float}(1))
end

function x_list_add(self::LSTM_Network, x::AFArray{gpu_float,1})
    push!(self.x_list,x)
    if length(self.x_list) > length(self.lstm_node_list)
        lstm_state=LSTM_state(self.lstm_param["mem_cell_ct"], self.lstm_param["x_dim"])
	push!(self.lstm_node_list, LSTM_Node(self.lstm_param,lstm_state,nothing,nothing,nothing,nothing))
    end
    idx = length(self.x_list) - 1
    if idx == 0
        # no recurrent inputs
        bottom_data_is(self.lstm_node_list[idx+1],x)
    else
        s_prev = self.lstm_node_list[idx+1].state["s"]
        h_prev = self.lstm_node_list[idx+1].state["h"]
        bottom_data_is(self.lstm_node_list[idx+1],x,s_prev,h_prev)
    end
end

const print_after_num_iters = 50
const iterations = 1000

function run_example()
    const mem_cell_ct = 100
    const x_dim = 50
    const concat_len = x_dim + mem_cell_ct
    lstm_param = LSTM_Param(mem_cell_ct, x_dim)
    lstm_net = LSTM_Network(lstm_param,Array{LSTM_Node}(1),Array{Any}(Array{gpu_float}(1)))
    const y_list_inp = [-0.5,0.2,0.1,-0.5]
    const y_list = AFArray{gpu_float,1}(map(gpu_float,y_list_inp))
    input_val_arr = [AFArray(map(gpu_float,rand(x_dim))) for i in y_list]
    for cur_iter in collect(1:1:iterations)
        print_=false 
        if cur_iter % print_after_num_iters == 1
            println("Iteration #$(cur_iter-1)")
            print_=true
        end
        for ind in collect(0:1:length(y_list)-1)
            x_list_add(lstm_net,input_val_arr[ind+1])
	    if ind==0
		deleteat!(lstm_net.lstm_node_list,1) # first reference is by default #undef
	    end
            if print_
                println("y_pred[$(ind+1)] :", lstm_net.lstm_node_list[ind+1].state["h"][1])
            end
        end
        loss = y_list_is(lstm_net,y_list,LossLayer)
        if print_
            println(loss)
        end
        apply_diff(lstm_param,0.1)
        x_list_clear(lstm_net)
    end
end

run_example()
