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
      //coord_type const& direction_,
      double weight_)
    :
      i_seqs(i_seqs_),
      eq_edge(eq_edge_),
      eq_edge_reverse(eq_edge_reverse_),
      d_edge(d_edge_),
      d_edge_reverse(d_edge_reverse_),
      //direction(direction_),
      weight(weight_)
    {
      MMTBX_ASSERT(weight >= 0);
    }
    
    den_lasso_proxy(
      i_seqs_type const& i_seqs_,
      coord_type const& eq_edge_,
      coord_type const& d_edge_,
      //coord_type const& direction_,
      double weight_)
    :
      i_seqs(i_seqs_),
      eq_edge(eq_edge_),
      eq_edge_reverse(eq_edge_),
      d_edge(d_edge_),
      d_edge_reverse(d_edge_),
      //direction(direction_),
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
      //direction(proxy.direction),
      weight(proxy.weight)
    {
      MMTBX_ASSERT(weight >= 0);
    }

    i_seqs_type i_seqs;
    coord_type eq_edge;
    coord_type eq_edge_reverse;
    coord_type d_edge;
    coord_type d_edge_reverse;
    //coord_type direction,
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
  den_update_weight(
    af::const_ref<scitbx::vec3<double> > const& sites_cart,
    af::ref<den_lasso_proxy> const& proxies,
    bool use_direction,
    bool use_scad=false,
    double scad_weight=3.0)
  {
    double a_scad_weight = 3.7*scad_weight;
    for (std::size_t i = 0; i < proxies.size(); i++){
      af::tiny<unsigned, 2> const& i_seqs = proxies[i].i_seqs;
      scitbx::vec3<double> dev;
      if(!use_direction){
        dev = sites_cart[ i_seqs[0] ] - sites_cart[ i_seqs[1] ];
        //dev += proxies[i].eq_edge - proxies[i].eq_edge_reverse;
      }
      else{
        dev = proxies[i].eq_edge - proxies[i].eq_edge_reverse;
      }
      if(!use_scad) proxies[i].weight = std::sqrt(1.24/dev.length());
      else{
        double dist = dev.length();
        //dist = dist/1.24 - 1;
        if(dist <= scad_weight){
          proxies[i].weight = 1.;
        }
        else{
          if(a_scad_weight < dist){
            proxies[i].weight = 0;
          }
          else{
            proxies[i].weight = (3.7 - dist/scad_weight)/2.7;
          }
        }
      }
    }
  }

  inline
  void
  den_update_edges_scad(
    af::const_ref<scitbx::vec3<double> > const& sites_cart,
    af::ref<den_lasso_proxy> const& proxies,
    double scad_weight,
    double den_weight,
    double penalty)
  {
    double c = den_weight;
    double lambda = scad_weight;
    double rho = penalty;
    for (std::size_t i = 0; i < proxies.size(); i++){
      den_lasso_proxy proxy = proxies[i];
      af::tiny<scitbx::vec3<double>, 2> sites;
      af::tiny<unsigned, 2> const& i_seqs = proxy.i_seqs;
      sites[0] = sites_cart[ i_seqs[0] ];
      sites[1] = sites_cart[ i_seqs[1] ];
      scitbx::vec3<double> devi = sites[0] + proxy.d_edge;//a
      scitbx::vec3<double> devj = sites[1] + proxy.d_edge_reverse;//b
      scitbx::vec3<double> dev = devi - devj;
      double theta;
      double dist = dev.length();//d
      double c_rho = c/rho;
      if(dist < 1e-20){//for numerical stability
        theta = 0.5;
      }
      else {
        double threshold = lambda/dist;
        if( threshold >= 1./(1.+2.*c_rho) ){
          theta = threshold*c_rho;
          theta = theta < 0.5 ? theta : 0.5;
        }
        else{
          if( threshold > 1./3.7 ){
            theta = (1. - 3.7*threshold)/(2. - rho*2.7/c);
          }
          else{
            theta = 0.;
          }
        }
      }
      //z_update
      proxies[i].eq_edge = (1-theta)*devi + theta*devj;
      proxies[i].eq_edge_reverse = theta*devi + (1-theta)*devj;
      //u_update
      proxies[i].d_edge += sites[0] - proxies[i].eq_edge;
      proxies[i].d_edge_reverse += sites[1] - proxies[i].eq_edge_reverse;
    }
  }


  inline
  double
  den_update_eq_edges(
    af::const_ref<scitbx::vec3<double> > const& sites_cart,
    af::ref<den_lasso_proxy> const& proxies,
    bool use_direction,
    double den_weight=1.0,
    double penalty=1.0)
  {
    double primal_resi = 0;
    scitbx::vec3<double> null_vec(0.,0.,0.);
    double factor = den_weight/penalty;
    af::shared<scitbx::vec3<double> > dual_sites(sites_cart.size(), null_vec);
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
      if(use_direction){
        //note that when direction penalty is used, the eq_edge stores the original
        //coordinates, while the eq_egde stores the shifts when direction is not used
        devi = devi - proxy.eq_edge;
        devj = devj - proxy.eq_edge_reverse;
        //dev = devi - devj;
        scitbx::vec3<double> direction = proxy.eq_edge - proxy.eq_edge_reverse;
        direction = direction.normalize();
        double prod_i = devi*direction;
        double prod_j = devj*direction;
        double prod = prod_i - prod_j;
        double threshold = 2*factor*proxy.weight;
        //z_update
        if(std::abs(prod) > threshold){
          if(prod > threshold){
            prod_i = 1;
            prod_j = -1;
          }
          else if(prod < -threshold){
            prod_i = -1;
            prod_j = 1;
          }
          scitbx::vec3<double> d_z = devi - (prod_i*factor*proxy.weight)*direction;
          proxies[i].eq_edge = proxies[i].eq_edge + d_z;
          d_z = devj + (prod_j*factor*proxy.weight)*direction;
          proxies[i].eq_edge_reverse = proxies[i].eq_edge_reverse + d_z;
        }
        else{
          prod_i = prod_i < 1 ? prod_i : 1;
          prod_i = prod_i > -1 ? prod_i : -1;
          prod_j = prod_i - prod;
          scitbx::vec3<double> d_z = devi - prod_i*direction;
          proxies[i].eq_edge = proxies[i].eq_edge + d_z;
          d_z = devj + prod_j*direction;
          proxies[i].eq_edge_reverse = proxies[i].eq_edge_reverse + d_z;
        }
      }
      else{
        if(dist < 1e-20){
          theta = 0.5;
        }
        else{
          theta = 1 - (factor/dist)*proxy.weight;
          //theta = 1 - factor/dist;
          theta = theta > 0.5 ? theta: 0.5;
        }
        //z_update
        scitbx::vec3<double> z_i = theta*devi + (1-theta)*devj;//z_k+1
        scitbx::vec3<double> z_j = theta*devj + (1-theta)*devi;
        //record z changes
        dual_sites[ i_seqs[0] ] += z_i - proxies[i].eq_edge;//z_k+1 - z_k
        dual_sites[ i_seqs[1] ] += z_j - proxies[i].eq_edge_reverse;
        proxies[i].eq_edge = z_i;
        proxies[i].eq_edge_reverse = z_j;

      }
      //u_update
      devi = sites[0] - proxies[i].eq_edge;
      devj = sites[1] - proxies[i].eq_edge_reverse;
      primal_resi += devi.length_sq() + devj.length_sq();
      proxies[i].d_edge += devi;
      proxies[i].d_edge_reverse += devj;
    }
    double dual_resi = 0;
    for(std::size_t i = 0; i < dual_sites.size(); i++){
      dual_resi += dual_sites[i].length_sq();
    }
    dual_resi = std::sqrt(dual_resi);
    primal_resi = std::sqrt(primal_resi);
    double rho = dual_resi/primal_resi;
    if(rho > 10*penalty) rho = 2*penalty;
    else if(rho < 0.1*penalty) rho = penalty/2;
    else rho = penalty;
    if (rho != penalty){
      factor = penalty/rho;
      for(std::size_t i = 0; i < proxies.size(); i++){
        proxies[i].d_edge *= rho;
      }
    }
    return rho;
  }

  inline
  void
  den_reset_eq_edges(
    af::const_ref<scitbx::vec3<double> > const& sites_cart,
    af::ref<den_lasso_proxy> const& proxies,
    bool use_scad,
    bool use_sqrt,
    bool use_direction,
    double scad_weight
    )
  {
    double a_scad_weight = scad_weight*3.7;
    for (std::size_t i = 0; i < proxies.size(); i++){
      af::tiny<unsigned, 2> const& i_seqs = proxies[i].i_seqs;
      scitbx::vec3<double> dev = sites_cart[ i_seqs[0] ] - sites_cart[ i_seqs[1] ];
      if(!use_scad && use_sqrt) proxies[i].weight = std::sqrt(1.24/dev.length());
      else if(use_scad) {
        double dist = dev.length();
        //dist = dist/1.24 - 1;
        if(dist <= scad_weight){
          proxies[i].weight = 1.;
        }
        else{
          if(a_scad_weight < dist){
            proxies[i].weight = 0;
          }
          else{
            proxies[i].weight = (3.7 - dist/scad_weight)/2.7;
          }
        }
      }
      else proxies[i].weight = 1.0;
      if(!use_direction){
        for(std::size_t j = 0; j < 3; j++){
          proxies[i].eq_edge[j] = 0;
          proxies[i].eq_edge_reverse[j] = 0;
          proxies[i].d_edge[j] = 0;
          proxies[i].d_edge_reverse[j] = 0;
        }
      }
      else{
        for(std::size_t j = 0; j < 3; j++){
          proxies[i].eq_edge[j] = sites_cart[ i_seqs[0] ][j];
          proxies[i].eq_edge_reverse[j] = sites_cart[ i_seqs[1] ][j];
          proxies[i].d_edge[j] = 0;
          proxies[i].d_edge_reverse[j] = 0;
        }
      }
    }
  }

}}
