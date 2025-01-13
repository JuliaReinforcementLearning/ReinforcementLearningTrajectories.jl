using Term

const TRACE_COLORS = (
    "bright_green",
    "hot_pink",
    "bright_blue",
    "light_coral",
    "bright_cyan",
    "sandy_brown",
    "violet",
)

Base.show(io::IO, ::MIME"text/plain", t::Union{Trace,Traces,Episode,Episodes,Trajectory}) =
    tprint(io, convert(Term.AbstractRenderable, t; width = displaysize(io)[2]) |> string)

inner_convert(::Type{Term.AbstractRenderable}, s::String; style = "gray1", width = 88) =
    Panel(s, width = width, style = style, justify = :center)
inner_convert(t::Type{Term.AbstractRenderable}, x::Union{Symbol,Number}; kw...) =
    inner_convert(t, string(x); kw...)

function inner_convert(
    ::Type{Term.AbstractRenderable},
    x::AbstractArray;
    style = "gray1",
    width = 88,
)
    t = string(nameof(typeof(x)))
    s = replace(string(size(x)), " " => "")
    Panel(t * "\n" * s, style = style, justify = :center, width = width)
end

function inner_convert(::Type{Term.AbstractRenderable}, x; style = "gray1", width = 88)
    s = string(nameof(typeof(x)))
    Panel(s, style = style, justify = :center, width = width)
end

Base.convert(T::Type{Term.AbstractRenderable}, t::Trace{<:AbstractArray}; kw...) = convert(
    T,
    Trace(collect(eachslice(t.x, dims = ndims(t.x))));
    kw...,
    type = typeof(t),
    subtitle = "size: $(size(t.x))",
)
Base.convert(T::Type{Term.AbstractRenderable}, t::NormalizedTrace; kw...) =
    convert(T, t.trace; kw..., type = typeof(t))
function Base.convert(
    ::Type{Term.AbstractRenderable},
    t::Trace{<:AbstractVector};
    width = 88,
    n_head = 2,
    n_tail = 1,
    name = "Trace",
    style = TRACE_COLORS[mod1(hash(name), length(TRACE_COLORS))],
    type = typeof(t),
    subtitle = "size: $(size(t.x))",
)
    title = "$name: [italic]$type[/italic] "
    min_width = min(width, length(title) - 4)

    n = length(t.x)
    if n == 0
        content = ""
    elseif 1 <= n <= n_head + n_tail
        content = mapreduce(
            x -> inner_convert(
                Term.AbstractRenderable,
                x,
                style = style,
                width = min_width - 6,
            ),
            /,
            t.x,
        )
    else
        content =
            mapreduce(
                x -> inner_convert(
                    Term.AbstractRenderable,
                    x,
                    style = style,
                    width = min_width - 6,
                ),
                /,
                t.x[1:n_head],
            ) / TextBox("...", justify = :center, width = min_width - 6) / mapreduce(
                x -> inner_convert(
                    Term.AbstractRenderable,
                    x,
                    style = style,
                    width = min_width - 6,
                ),
                /,
                t.x[end-n_tail+1:end],
            )
    end
    Panel(
        content,
        width = min_width,
        title = title,
        subtitle = subtitle,
        subtitle_justify = :right,
        style = style,
        subtitle_style = "yellow",
    )
end

function Base.convert(::Type{Term.AbstractRenderable}, t::Traces; width = 88)
    max_len = mapreduce(length, max, t.traces)
    min_len = mapreduce(length, min, t.traces)
    if max_len - min_len == 1
        n_tails = [length(x) == max_len ? 2 : 1 for x in t.traces]
    else
        n_tails = [1 for x in t.traces]
    end
    N = length(t.traces)
    max_inner_width = ceil(Int, (width - 6 * 2) / N)
    Panel(
        mapreduce(
            ((i, x),) -> convert(
                Term.AbstractRenderable,
                t[x];
                width = max_inner_width,
                name = x,
                n_tail = n_tails[i],
                style = TRACE_COLORS[mod1(i, length(TRACE_COLORS))],
            ),
            *,
            enumerate(keys(t)),
        ),
        title = "Traces",
        style = "yellow3",
        subtitle = "$N traces in total",
        subtitle_justify = :right,
        width = width,
        fit = true,
    )
end

function Base.convert(::Type{Term.AbstractRenderable}, e::Episode; width = 88)
    Panel(
        convert(Term.AbstractRenderable, e.traces; width = width - 6),
        title = "Episode",
        style = "green_yellow",
        subtitle = e[] ? "Episode END" : "Episode growing...",
        subtitle_justify = :right,
        width = width,
        fit = true,
    )
end

function Base.convert(::Type{Term.AbstractRenderable}, e::Episodes; width = 88)
    n = length(e)
    if n == 0
        content = ""
    elseif n == 1
        content = convert(Term.AbstractRenderable, e[1], width = width - 6)
    elseif n == 2
        content =
            convert(Term.AbstractRenderable, e[1], width = width - 6) /
            convert(Term.AbstractRenderable, e[end], width = width - 6)
    else
        content =
            convert(Term.AbstractRenderable, e[1], width = width - 6) /
            TextBox("...", justify = :center, width = width - 6) /
            convert(Term.AbstractRenderable, e[end], width = width - 6)
    end

    Panel(
        content,
        title = "Episodes",
        subtitle = "$n episodes in total",
        subtitle_justify = :right,
        width = width,
        fit = true,
        style = "wheat1",
    )
end

function Base.convert(r::Type{Term.AbstractRenderable}, t::Trajectory; width = 88)
    Panel(
        convert(r, t.container; width = width - 8) / Panel(
            convert(Term.Tree, t.sampler);
            title = "sampler",
            style = "yellow3",
            fit = true,
            width = width - 8,
        ) / Panel(
            convert(Term.Tree, t.controller);
            title = "controller",
            style = "yellow3",
            fit = true,
            width = width - 8,
        );
        title = "Trajectory",
        style = "yellow3",
        width = width,
        fit = true,
    )
end

# general converter

Base.convert(::Type{Term.Tree}, x) = Tree(to_tree_body(x); title = to_tree_title(x))
Base.convert(::Type{Term.Tree}, x::Tree) = x

function to_tree_body(x)
    pts = propertynames(x)
    if length(pts) > 0
        Dict(
            "$p => $(summary(getproperty(x, p)))" => to_tree_body(getproperty(x, p)) for
            p in pts
        )
    else
        x
    end
end

to_tree_title(x) = "$(summary(x))"
