function rand_arr(a::Float64,b::Float64,dims::Int...)
    srand(0)
    arr = zeros(Float64,dims)
    rand!(arr)
    arr*(b-a)+a
end

function sigmoid_helper(x::Float64)
    return 1/(1+exp(-x))
end

function sigmoid(x::Union{Array{Float64,1},Float64})
    try
        return sigmoid_helper(x)
    catch
        return map(sigmoid_helper,x)
    end
end

function mat_dot(a::Union{Array{Float64,1},Array{Float64,2}},b::Array{Float64,1})
    try
        return dot(a,b)
    catch
        *(a,b)
    end
end

function outer(a::Array{Float64,1},b::Array{Float64,1})
    *(vec(a),transpose(vec(b)))
end

function dot_transpose(a::Union{Array{Float64,1},Array{Float64,2}},b::Array{Float64,1})
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
         "wg_diff"=> zeros(Float64,(mem_cell_ct, concat_len)),
         "wi_diff"=> zeros(Float64,(mem_cell_ct, concat_len)),
         "wf_diff"=> zeros(Float64,(mem_cell_ct, concat_len)),
         "wo_diff"=> zeros(Float64,(mem_cell_ct, concat_len)),
         "bg_diff"=> zeros(Float64,mem_cell_ct),
         "bi_diff"=> zeros(Float64,mem_cell_ct),
         "bf_diff"=> zeros(Float64,mem_cell_ct),
         "bo_diff"=> zeros(Float64,mem_cell_ct),
         "concat_len"=> concat_len,
         "mem_cell_ct"=> mem_cell_ct,
         "x_dim"=> x_dim
    )
end

function apply_diff(params_dict::Dict{String,Any}, lr::Float64)
    params = params_dict
    
    params["wg"] -= lr*params["wg_diff"]
    params["wi"] -= lr*params["wi_diff"]
    params["wf"] -= lr*params["wf_diff"]
    params["wo"] -= lr*params["wo_diff"]
    params["bg"] -= lr*params["bg_diff"]
    params["bi"] -= lr*params["bi_diff"]
    params["bf"] -= lr*params["bf_diff"]
    params["bo"] -= lr*params["bo_diff"]

    params["wg_diff"] = zeros(size(params["wg"]))
    params["wi_diff"] = zeros(size(params["wi"])) 
    params["wf_diff"] = zeros(size(params["wf"])) 
    params["wo_diff"] = zeros(size(params["wo"])) 
    params["bg_diff"] = zeros(size(params["bg"]))
    params["bi_diff"] = zeros(size(params["bi"])) 
    params["bf_diff"] = zeros(size(params["bf"])) 
    params["bo_diff"] = zeros(size(params["bo"])) 
    params
end

abstract LSTM

function LSTM_state(mem_cell_ct::Int, x_dim::Int)
    Dict("g"=> zeros(mem_cell_ct),
         "i"=> zeros(mem_cell_ct),
         "f"=> zeros(mem_cell_ct),
         "o"=> zeros(mem_cell_ct),
         "s"=> zeros(mem_cell_ct),
         "h"=> zeros(mem_cell_ct),
         "bottom_diff_h"=> zeros(mem_cell_ct),
         "bottom_diff_s"=> zeros(mem_cell_ct),
         "bottom_diff_x"=> zeros(x_dim),
         "mem_cell_ct"=> mem_cell_ct,
         "x_dim"=> x_dim)
end

type LSTM_Node <: LSTM
    param
    state
    x
    xc
    s_prev
    h_prev
end

function bottom_data_is(self::LSTM_Node,x::Array{Float64,1},s_prev=nothing::Union{typeof(nothing),Array},h_prev=nothing::Union{typeof(nothing),Array})
    if s_prev==nothing
        s_prev = zeros(self.state["s"])
    end
    if h_prev==nothing
        h_prev = zeros(self.state["h"])
    end
    self.s_prev=vec(s_prev)
    self.h_prev=vec(h_prev)
    
    xc = vec(vcat(vec(x), vec(h_prev)))
        
    self.state["g"] = tanh(mat_dot(self.param["wg"],xc)+self.param["bg"])
    self.state["i"] = sigmoid(mat_dot(self.param["wi"],xc)+self.param["bi"])
    self.state["f"] = sigmoid(mat_dot(self.param["wf"],xc)+self.param["bf"])
    self.state["o"] = sigmoid(mat_dot(self.param["wo"],xc)+self.param["bo"])
    self.state["s"] = self.state["g"].*self.state["i"]+s_prev.*self.state["f"]
    self.state["h"] = self.state["s"].*self.state["o"]
    self.x = vec(x)
    self.xc = vec(xc)
    
    LSTM_Node
end

function top_diff_is(self::LSTM_Node,top_diff_h::Array{Float64,1},top_diff_s::Array{Float64,1})
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
    
    dxc = zeros(size(self.xc))
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

function loss_func(self::Type{LossLayer}, pred::Array{Float64,1}, label::Float64)
    (pred[1]-label)^2 #squared error
end

function bottom_diff(self::Type{LossLayer}, pred::Array{Float64,1}, label::Float64)
    diff = zeros(pred)
    diff[1] = 2*(pred[1]-label)
    diff
end

type LSTM_Network <: LSTM
    lstm_param
    lstm_node_list
    x_list
end


function y_list_is(self::LSTM_Network,y_list::Array{Float64,1},loss_layer::Type{LossLayer})
    if length(y_list) != length(self.x_list)
        error("AssertionError: length(y_list) is not equal to length(self.x_list)")
    end
    idx = length(self.x_list)-1
    loss = loss_func(loss_layer, self.lstm_node_list[idx+1].state["h"], y_list[idx+1])
    diff_h = bottom_diff(loss_layer, self.lstm_node_list[idx+1].state["h"], y_list[idx+1])
    diff_s = zeros(self.lstm_param["mem_cell_ct"])
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
    self.x_list=[]
end

function x_list_add(self::LSTM_Network, x::Array{Float64,1})
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
    mem_cell_ct = 100
    x_dim = 50
    concat_len = x_dim + mem_cell_ct
    lstm_param = LSTM_Param(mem_cell_ct, x_dim)
    lstm_net = LSTM_Network(lstm_param,[],[])
    y_list = [-0.5,0.2,0.1,-0.5]
    input_val_arr = [rand(x_dim) for i in y_list]
    for cur_iter in collect(1:1:iterations)
        print_=false 
        if cur_iter % print_after_num_iters == 1
            println("Iteration #$(cur_iter-1)")
            print_=true
        end
        for ind in collect(0:1:length(y_list)-1)
            x_list_add(lstm_net,input_val_arr[ind+1])
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
