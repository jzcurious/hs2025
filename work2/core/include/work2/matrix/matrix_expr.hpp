#ifndef _MATRIX_EXPR_HPP_
#define _MATRIX_EXPR_HPP_

#include "work2/matrix/matrix_view_kind.hpp"
#include <tuple>

template <class FuncT, class ResT, class... ArgTs>
concept MatrixExprTemplateArgs = requires {
  MatrixViewKind<ResT>;
  (MatrixViewKind<ArgTs> and ...);
  std::invocable<FuncT, ResT, ArgTs...>;
};

template <class FuncT, MatrixViewKind ResT, MatrixViewKind... ArgTs>
  requires(MatrixExprTemplateArgs<FuncT, ResT, ArgTs...>)
struct MatrixExpr final {
 private:
  FuncT _func;
  std::tuple<ArgTs...> _args;

 public:
  template <class FuncT_, class... ArgTs_>
  MatrixExpr(FuncT_&& func, ArgTs_&&... args)
      : _func(std::forward<FuncT_>(func))
      , _args(std::forward<ArgTs_>(args)...) {}

  ResT& eval(ResT& res) {
    return std::apply(
        [this, &res](
            auto&&... args) { _func(res, std::forward<decltype(args)>(args)...); },
        _args);
  }
};

#endif  // _MATRIX_EXPR_HPP_
