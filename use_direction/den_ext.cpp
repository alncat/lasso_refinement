
#include <cctbx/boost_python/flex_fwd.h>

#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
#include <boost/python/class.hpp>
#include <boost/python/args.hpp>
#include <boost/python/return_value_policy.hpp>
#include <boost/python/return_by_value.hpp>
#include <boost/optional.hpp>

#include <mmtbx/den/den.h>
#include <cctbx/geometry_restraints/proxy_select.h>
#include <scitbx/array_family/boost_python/shared_wrapper.h>

namespace mmtbx { namespace den {
namespace {

  void wrap_simple_restraints()
  {
    using namespace boost::python;
    typedef return_value_policy<return_by_value> rbv;
    typedef den_simple_proxy w_t;
    typedef den_lasso_proxy l_t;
    class_<w_t>("den_simple_proxy", no_init)
      .def(init<
        af::tiny<unsigned, 2> const&, double, double, double >((
          arg("i_seqs"),
          arg("eq_distance"),
          arg("eq_distance_start"),
          arg("weight"))))
      .add_property("i_seqs", make_getter(&w_t::i_seqs, rbv()))
      .def_readwrite("eq_distance", &w_t::eq_distance)
      .def_readwrite("eq_distance_start", &w_t::eq_distance_start)
      .def_readwrite("weight", &w_t::weight)
    ;
    {
      typedef return_internal_reference<> rir;
      scitbx::af::boost_python::shared_wrapper<den_simple_proxy, rir>::wrap(
        "shared_den_simple_proxy")
        .def("proxy_select",
          (af::shared<w_t>(*)(
           af::const_ref<w_t> const&,
           std::size_t,
           af::const_ref<std::size_t> const&))
           cctbx::geometry_restraints::shared_proxy_select, (
         arg("n_seq"), arg("iselection")));
    }

    class_<l_t>("den_lasso_proxy", no_init)
      .def(init<
        af::tiny<unsigned, 2> const&, 
        scitbx::vec3<double> const&, 
        scitbx::vec3<double> const&, 
        scitbx::vec3<double> const&,
        scitbx::vec3<double> const&,
        double >((
          arg("i_seqs"),
          arg("eq_edge"),
          arg("eq_edge_reverse"),
          arg("d_edge"),
          arg("d_edge_reverse"),
          arg("weight"))))
      .add_property("i_seqs", make_getter(&l_t::i_seqs, rbv()))
      .add_property("eq_edge", make_getter(&l_t::eq_edge, rbv()))
      .add_property("eq_edge_reverse", make_getter(&l_t::eq_edge_reverse, rbv()))
      .add_property("d_edge_reverse", make_getter(&l_t::d_edge_reverse, rbv()))
      .add_property("d_edge", make_getter(&l_t::d_edge, rbv()))
      .def_readwrite("weight", &l_t::weight)
    ;
    {
      typedef return_internal_reference<> rir;
      scitbx::af::boost_python::shared_wrapper<den_lasso_proxy, rir>::wrap(
        "shared_den_lasso_proxy")
        .def("proxy_select",
          (af::shared<l_t>(*)(
           af::const_ref<l_t> const&,
           std::size_t,
           af::const_ref<std::size_t> const&))
           cctbx::geometry_restraints::shared_proxy_select, (
        arg("n_seq"), arg("iselection")));
    }

    def("den_simple_residual_sum",
      (double(*)(
        af::const_ref<scitbx::vec3<double> > const&,
        af::const_ref<den_simple_proxy> const&,
        af::ref<scitbx::vec3<double> > const&,
        double,
        double))
      den_simple_residual_sum, (
      arg("sites_cart"),
      arg("proxies"),
      arg("gradient_array"),
      arg("den_weight")=1.0));

    def("den_lasso_residual_sum",
      (double(*)(
        af::const_ref<scitbx::vec3<double> > const&,
        af::const_ref<den_lasso_proxy> const&,
        af::ref<scitbx::vec3<double> > const&,
        double,
        double))
      den_lasso_residual_sum, (
      arg("sites_cart"),
      arg("proxies"),
      arg("gradient_array"),
      arg("penalty")=1.0));

    def("den_update_eq_distances",
    (void(*)(
      af::const_ref<scitbx::vec3<double> > const&,
      af::ref<den_simple_proxy> const&,
      double,
      double))
    den_update_eq_distances, (
    arg("sites_cart"),
    arg("proxies"),
    arg("gamma"),
    arg("kappa")));

    def("den_update_weight",
      (void(*)(
        af::const_ref<scitbx::vec3<double> > const&,
        af::ref<den_lasso_proxy> const&,
        bool,
        bool,
        double,
        double))
      den_update_weight, (
      arg("sites_cart"),
      arg("proxies"),
      arg("use_direction"),
      arg("use_scad"),
      arg("scad_weight")=3.0));

    def("den_update_eq_edges",
    (double(*)(
      af::const_ref<scitbx::vec3<double> > const&,
      af::ref<den_lasso_proxy> const&,
      bool,
      double,
      double,
      double,
      double))
    den_update_eq_edges, (
    arg("sites_cart"),
    arg("proxies"),
    arg("use_direction"),
    arg("den_weight")=1.0,
    arg("penalty")=1.0));

    def("den_update_edges_scad",
    (void(*)(
      af::const_ref<scitbx::vec3<double> > const&,
      af::ref<den_lasso_proxy> const&,
      double,
      double,
      double))
    den_update_edges_scad, (
    arg("sites_cart"),
    arg("proxies"),
    arg("scad_weight"),
    arg("den_weight"),
    arg("penalty")));

    def("den_reset_eq_edges",
    (void(*)(
      af::const_ref<scitbx::vec3<double> > const&,
      af::ref<den_lasso_proxy> const&,
      bool,
      bool,
      bool,
      double
      ))
    den_reset_eq_edges, (
    arg("sites_cart"),
    arg("proxies"),
    arg("use_scad"),
    arg("use_sqrt"),
    arg("use_direction"),
    arg("scad_weight")));

  }
} // namespace anonymous

namespace boost_python {

  void wrap_den() {
    wrap_simple_restraints();
  }

} // namespace boost_python

}} //namespace mmtbx::den

BOOST_PYTHON_MODULE(mmtbx_den_restraints_ext)
{
  mmtbx::den::boost_python::wrap_den();
}

