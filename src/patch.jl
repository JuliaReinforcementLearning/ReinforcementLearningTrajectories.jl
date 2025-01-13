#####

import StackViews: StackView

lazy_stack(x) = StackView(x)
lazy_stack(x::AbstractVector{<:Number}) = x
