#ifndef _MATRIX_EXPR_HPP_
#define _MATRIX_EXPR_HPP_

#include "work2/matrix/matrix_view_kind.hpp"
#include <tuple>

template <class FuncT, class ResT, class... ArgTs>
concept MatrixExprTemplateArgs = MatrixViewKind<ResT> and (MatrixViewKind<ArgTs> and ...)
                                 and std::invocable<FuncT, ResT&, ArgTs...>;

template <class FuncT, MatrixViewKind ResT, MatrixViewKind... ArgTs>
  requires(MatrixExprTemplateArgs<FuncT, ResT, ArgTs...>)
struct MatrixExpr final {
 private:
  FuncT _func;
  ResT _res;
  std::tuple<ArgTs...> _args;

 public:
  using result_t = ResT;

  struct matrix_expr_f {};

  template <class FuncT_, class ResT_, class... ArgTs_>
  MatrixExpr(FuncT_&& func, ResT_& res, ArgTs_&&... args)
      : _func(std::forward<FuncT_>(func))
      , _res(res)
      , _args(std::forward<ArgTs_>(args)...) {}

  ResT& eval(ResT& res) {
    std::apply([this, &res](
                   auto&&... args) { _func(res, std::forward<decltype(args)>(args)...); },
        _args);
    return res;
  }

  ResT& required_result_view() {
    return _res;
  }
};

#endif  // _MATRIX_EXPR_HPP_
