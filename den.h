#include <mmtbx/error.h>
#include <cctbx/geometry_restraints/bond.h>
#include <cctbx/geometry/geometry.h>
#include <cmath>
#include <set>
#include <iostream>

namespace mmtbx { namespace den {

  namespace af = scitbx::af;
  using cctbx::geometry_restraints::bond;
  using cctbx::geometry_restraints::bond_simple_proxy;

  struct den_simple_proxy : bond_simple_proxy
  {
    typedef af::tiny<unsigned, 2> i_seqs_type;

    den_simple_proxy() {}

    den_simple_proxy(
      i_seqs_type const& i_seqs_,
      double eq_distance_,
      double eq_distance_start_,
      double weight_)
    :
      i_seqs(i_seqs_),
      eq_distance(eq_distance_),
      eq_distance_start(eq_distance_start_),
      weight(weight_)
    {
      MMTBX_ASSERT((eq_distance > 0) && (eq_distance_start > 0));
    }

    // Support for proxy_select (and similar operations)
    den_simple_proxy(
      i_seqs_type const& i_seqs_,
      den_simple_proxy const& proxy)
    :
      i_seqs(i_seqs_),
      eq_distance(proxy.eq_distance),
      eq_distance_start(proxy.eq_distance_start),
      weight(proxy.weight)
    {
      MMTBX_ASSERT((eq_distance > 0) && (eq_distance_start > 0));
    }

    i_seqs_type i_seqs;
    double eq_distance;
    double eq_distance_start;
    double weight;
  };

  struct den_lasso_proxy
  {
    typedef af::tiny<unsigned, 2> i_seqs_type;
    typedef scitbx::vec3<double> coord_type;

    den_lasso_proxy() {}
    
    den_lasso_proxy(
      i_seqs_type const& i_seqs_,
      coord_type const& eq_edge_,
      coord_type const& eq_edge_reverse_,
      coord_type const& d_edge_,
      coord_type const& d_edge_reverse_,
      double weight_)
    :
      i_seqs(i_seqs_),
      eq_edge(eq_edge_),
      eq_edge_reverse(eq_edge_reverse_),
      d_edge(d_edge_),
      d_edge_reverse(d_edge_reverse_),
      weight(weight_)
    {
      MMTBX_ASSERT(weight >= 0);
    }
    
    den_lasso_proxy(
      i_seqs_type const& i_seqs_,
      coord_type const& eq_edge_,
      coord_type const& d_edge_,
      double weight_)
    :
      i_seqs(i_seqs_),
      eq_edge(eq_edge_),
      eq_edge_reverse(eq_edge_),
      d_edge(d_edge_),
      d_edge_reverse(d_edge_),
      weight(weight_)
    {
      MMTBX_ASSERT(weight >= 0);
    }

    den_lasso_proxy(
      i_seqs_type const& i_seqs_,
      den_lasso_proxy const& proxy)
    :
      i_seqs(i_seqs_),
      eq_edge(proxy.eq_edge),
      eq_edge_reverse(proxy.eq_edge_reverse),
      d_edge(proxy.d_edge),
      d_edge_reverse(proxy.d_edge_reverse),
      weight(proxy.weight)
    {
      MMTBX_ASSERT(weight >= 0);
    }

    i_seqs_type i_seqs;
    coord_type eq_edge;
    coord_type eq_edge_reverse;
    coord_type d_edge;
    coord_type d_edge_reverse;
    double weight;
  };


  inline
  double
  den_simple_residual_sum(
    af::const_ref<scitbx::vec3<double> > const& sites_cart,
    af::const_ref<den_simple_proxy> const& proxies,
    af::ref<scitbx::vec3<double> > const& gradient_array,
    double den_weight=1.0)
  {
    double residual_sum = 0;
    double slack = 0.0;
    unsigned n_sites = sites_cart.size();
    for (std::size_t i = 0; i < proxies.size(); i++) {
      den_simple_proxy proxy = proxies[i];
      af::tiny<scitbx::vec3<double>, 2> sites;
      af::tiny<unsigned, 2> const& i_seqs = proxy.i_seqs;
      sites[0] = sites_cart[ i_seqs[0] ];
      sites[1] = sites_cart[ i_seqs[1] ];
      MMTBX_ASSERT((i_seqs[0] < n_sites) && (i_seqs[1] < n_sites));
      //bond restraint(sites, proxy.eq_distance, proxy.weight, slack);
      bond restraint(sites, proxy.eq_distance, den_weight, slack);
      double residual = restraint.residual();
      //double grad_factor = den_weight;
      residual_sum += residual;
      if (gradient_array.size() != 0) {
        af::tiny<scitbx::vec3<double>, 2> gradients = restraint.gradients();
        //weight is now handled at the residual level
        gradient_array[ i_seqs[0] ] += gradients[0];// * grad_factor;
        gradient_array[ i_seqs[1] ] += gradients[1];// * grad_factor;
      }
    }
    return residual_sum;
  }

  inline
  double
  den_lasso_residual_sum(
    af::const_ref<scitbx::vec3<double> > const& sites_cart,
    af::const_ref<den_lasso_proxy> const& proxies,
    af::ref<scitbx::vec3<double> > const& gradient_array,
    double penalty=1.0)
  {
    double residual_sum = 0;
    double h_penalty = penalty/2.;
    unsigned n_sites = sites_cart.size();
    for (std::size_t i = 0; i < proxies.size(); i++) {
      den_lasso_proxy proxy = proxies[i];
      af::tiny<scitbx::vec3<double>, 2> sites;
      af::tiny<unsigned, 2> const& i_seqs = proxy.i_seqs;
      sites[0] = sites_cart[ i_seqs[0] ];
      sites[1] = sites_cart[ i_seqs[1] ];
      MMTBX_ASSERT((i_seqs[0] < n_sites) && (i_seqs[1] < n_sites));
      //bond restraint(sites, proxy.eq_distance, proxy.weight, slack);
      double residual = 0;
      for(std::size_t j = 0; j < 3; j++){
        double d_s_d_e = sites[0][j] - proxy.eq_edge[j] + proxy.d_edge[j];
        double d_s_d_e_r = sites[1][j] - proxy.eq_edge_reverse[j] + proxy.d_edge_reverse[j];
        residual += h_penalty*d_s_d_e*d_s_d_e;
        residual += h_penalty*d_s_d_e_r*d_s_d_e_r;
        if(gradient_array.size() != 0) {
          gradient_array[i_seqs[0]][j] += penalty*d_s_d_e;
          gradient_array[i_seqs[1]][j] += penalty*d_s_d_e_r;
        }
      }
      //double grad_factor = den_weight;
      residual_sum += residual;
    }
    return residual_sum;
  }

  inline
  void
  den_update_eq_distances(
    af::const_ref<scitbx::vec3<double> > const& sites_cart,
    af::ref<den_simple_proxy> const& proxies,
    double gamma,
    double kappa)
  {
    double slack = 0.0;
    for (std::size_t i = 0; i < proxies.size(); i++) {
      den_simple_proxy proxy = proxies[i];
      af::tiny<scitbx::vec3<double>, 2> sites;
      af::tiny<unsigned, 2> const& i_seqs = proxy.i_seqs;
      sites[0] = sites_cart[ i_seqs[0] ];
      sites[1] = sites_cart[ i_seqs[1] ];
      bond restraint(sites, proxy.eq_distance, proxy.weight, slack);
      double distance_model = restraint.distance_model;
      double new_eq_dist = ((1.0-kappa)*proxy.eq_distance) +
                           ( kappa *
                             ( (gamma*distance_model) +
                               (1.0-gamma)*proxy.eq_distance_start ));
      proxies[i].eq_distance = new_eq_dist;
    }
  }
  
  inline
  void
  den_update_eq_edges(
    af::const_ref<scitbx::vec3<double> > const& sites_cart,
    af::ref<den_lasso_proxy> const& proxies,
    double den_weight=1.0,
    double penalty=1.0)
  {
    double factor = den_weight/penalty;
    for (std::size_t i = 0; i < proxies.size(); i++) {
      den_lasso_proxy proxy = proxies[i];
      af::tiny<scitbx::vec3<double>, 2> sites;
      af::tiny<unsigned, 2> const& i_seqs = proxy.i_seqs;
      sites[0] = sites_cart[ i_seqs[0] ];
      sites[1] = sites_cart[ i_seqs[1] ];
      scitbx::vec3<double> devi = sites[0] + proxy.d_edge;
      scitbx::vec3<double> devj = sites[1] + proxy.d_edge_reverse;
      scitbx::vec3<double> dev = devi - devj;
      double dist = dev.length();
      double theta;
      if(dist < 1e-20){
        theta = 0.5;
      }
      else{
        theta = 1 - (factor/dist)*proxy.weight;
        //theta = 1 - factor/dist;
        theta = theta > 0.5 ? theta: 0.5;
      }

      //z_update
      proxies[i].eq_edge = theta*devi + (1-theta)*devj;
      proxies[i].eq_edge_reverse = (1-theta)*devi + theta*devj;
      //u_update
      proxies[i].d_edge += sites[0] - proxies[i].eq_edge;
      proxies[i].d_edge_reverse += sites[1] - proxies[i].eq_edge_reverse;
    }
  }

  inline
  void
  den_reset_eq_edges(
    af::const_ref<scitbx::vec3<double> > const& sites_cart,
    af::ref<den_lasso_proxy> const& proxies
    )
  {
    for (std::size_t i = 0; i < proxies.size(); i++){
      af::tiny<unsigned, 2> const& i_seqs = proxies[i].i_seqs;
      scitbx::vec3<double> dev = sites_cart[ i_seqs[0] ] - sites_cart[ i_seqs[1] ];
      proxies[i].weight = std::sqrt(std::sqrt(1.55/dev.length()));
      for(std::size_t j = 0; j < 3; j++){
        proxies[i].eq_edge[j] = 0;
        proxies[i].eq_edge_reverse[j] = 0;
        proxies[i].d_edge[j] = 0;
        proxies[i].d_edge_reverse[j] = 0;
      }
    }
  }

}}
