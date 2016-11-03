from __future__ import division
from mmtbx.refinement import print_statistics
import mmtbx.utils
from libtbx.utils import Sorry
from libtbx import Auto

def set_data_target_type(params, f_obs, r_free_flags, hl_coeffs, model):
  result = params.main.target
  if(params.main.target.lower() == "auto"):
    if(params.main.min_number_of_test_set_reflections_for_max_likelihood_target>
       r_free_flags.data().count(True)):
      result = "ls"
    elif((params.main.use_experimental_phases is None or
        params.main.use_experimental_phases) and hl_coeffs is not None):
      result = "mlhl"
    else:
      result = "ml"
  if(params.main.target == "mlhl"):
    potential_non_zero_f_double_prime = False
    if(model.anomalous_scatterer_groups is not None):
      for group in model.anomalous_scatterer_groups:
        if(group.f_double_prime != 0 or group.refine_f_double_prime):
          potential_non_zero_f_double_prime = True
    if(f_obs.anomalous_flag() and potential_non_zero_f_double_prime):
      raise Sorry(
        "mlhl target function (maximum likelihood with experimental phases)"
        " currently not supported in combination with anomalous scattering:\n"
        "  Please use refinement.main.target=ml (you may have to disable\n"
        "  automatic adjustments; check previous messages) or specify\n"
        "  force_anomalous_flag_to_be_equal_to=False and set all\n"
        "  f_double_prime=0, refine_f_double_prime=False.")
  elif(params.main.target == "ml_sad"):
    if(f_obs.sigmas() is None):
      raise Sorry(
        "F-obs sigmas required but not available:\n"
        "  The\n"
        '    refinement.main.target="ml_sad"\n'
        "  target function requires the sigmas associated with F-obs.\n"
        "  Please provide a reflection file with F-obs"
          " and associated sigmas or\n"
        "  select another target function (e.g. refinement.main.target=ml).")
    if(not f_obs.anomalous_flag()):
      raise Sorry(
        "F-obs not anomalous:\n"
        "  The\n"
        '    refinement.main.target="ml_sad"\n'
        "  target function requires anomalous F-obs.\n"
        "  Please provide a reflection file with anomalous data or\n"
        "  select another target function (e.g. refinement.main.target=ml).")
  elif(params.main.target == "ls"):
    result = params.ls_target_names.target_name
  if(result.lower() == "auto"): result = "ml" # must be next to last line
  return result

def set_outliers_to_reference(
      params,
      fmodels,
      model,
      prefix,
      macro_cycle,
      reference_model_manager,
      log):
  if(reference_model_manager is not None):
    print_statistics.make_sub_header(prefix, out=log)
    if reference_model_manager.params.fix_outliers:
      reference_model_manager.set_rotamer_to_reference(
      xray_structure=model.xray_structure,
      log=log)
    pre_correct_r_values_string = "r_work= %.4f   r_free= %.4f" % (
      fmodels.fmodel_xray().r_work(),
      fmodels.fmodel_xray().r_free())
    fmodels.update_xray_structure(
      xray_structure = model.xray_structure,
      update_f_calc  = True,
      update_f_mask  = params.bulk_solvent_and_scale.bulk_solvent)
    print >> log, "Before rotamer outlier correction:"
    print >> log, pre_correct_r_values_string
    print >> log, "After rotamer outlier correction:"
    print >> log, "r_work= %.4f   r_free= %.4f" % (
      fmodels.fmodel_xray().r_work(),
      fmodels.fmodel_xray().r_free())

def ncs_rotamer_update (
      params,
      fmodels,
      model,
      macro_cycle,
      log):
  ncs_manager = \
    model.restraints_manager.geometry.generic_restraints_manager.ncs_manager
  if(ncs_manager is not None):
    if (ncs_manager.params.fix_outliers is Auto and \
        fmodels.fmodel_xray().f_obs().d_min() <= 3.0) or \
        (ncs_manager.params.fix_outliers is True):
      ncs_manager.fmodel = fmodels.fmodel_xray()
      ncs_manager.fix_rotamer_outliers(
        xray_structure=model.xray_structure,
        geometry_restraints_manager=model.restraints_manager.geometry,
        pdb_hierarchy=model.pdb_hierarchy(),
        log=log)
      if ncs_manager.last_round_outlier_fixes > 0:
        fmodels.update_xray_structure(
          xray_structure = model.xray_structure,
          update_f_calc  = True,
          update_f_mask  = params.bulk_solvent_and_scale.bulk_solvent)
    if (ncs_manager.params.check_rotamer_consistency is Auto and \
        fmodels.fmodel_xray().f_obs().d_min() <= 3.0) or \
        (ncs_manager.params.check_rotamer_consistency is True):
      ncs_manager.fmodel = fmodels.fmodel_xray()
      ncs_manager.fix_rotamer_consistency(
        xray_structure=model.xray_structure,
        geometry_restraints_manager=model.restraints_manager.geometry,
        pdb_hierarchy=model.pdb_hierarchy(sync_with_xray_structure=True),
        log=log)
      if ncs_manager.last_round_rotamer_changes > 0:
        fmodels.update_xray_structure(
          xray_structure = model.xray_structure,
          update_f_calc  = True,
          update_f_mask  = params.bulk_solvent_and_scale.bulk_solvent)

def highest_peaks_and_deepest_holes_in_residual_map(params, fmodels, model, log):
  main = params.main
  # XXX twin_fmodel need a fix
  if(main.show_residual_map_peaks_and_holes and params.twinning.twin_law is None):
    fms = fmodels.fmodel_xray(), fmodels.fmodel_neutron()
    for i_seq, fm in enumerate(fms):
      if(fms[1] is not None and i_seq == 0):
        print_statistics.make_sub_header("xray datat", out = log)
      if(fms[1] is not None and i_seq == 1):
        print_statistics.make_sub_header("neutron datat", out = log)
      if(fm is not None):
        import mmtbx.find_peaks
        mmtbx.find_peaks.show_highest_peaks_and_deepest_holes(
          fmodel = fm,
          pdb_atoms = model.pdb_atoms,
          map_type = "mFobs-DFmodel",
          map_cutoff_plus = 3.5,
          map_cutoff_minus = -3.5,
          log = log)

def den(
      params,
      model,
      fmodels,
      target_weights,
      macro_cycle,
      prefix,
      log):
  if hasattr(model.restraints_manager, 'geometry'):
    ncs_manager = model.restraints_manager.geometry.\
                    generic_restraints_manager.ncs_manager
  else:
    ncs_manager = None
  print_statistics.make_header(prefix, out=log)
  if not params.den.refine_lasso:
    from mmtbx.den import refinement as den_refinement
    den_refinement_manager = den_refinement.manager(
      fmodels=fmodels,
      model=model,
      params=params,
      target_weights=target_weights,
      macro_cycle=macro_cycle,
      ncs_manager=ncs_manager,
      log=log)
    if params.den.verbose:
      model.restraints_manager.geometry.\
        generic_restraints_manager.den_manager.\
        show_den_summary(sites_cart=model.xray_structure.sites_cart())
  else:
    from mmtbx.den import lasso_refinement as den_refinement
    den_refinement_manager = den_refinement.manager(
      fmodels=fmodels,
      model=model,
      params=params,
      target_weights=target_weights,
      macro_cycle=macro_cycle,
      ncs_manager=ncs_manager,
      log=log)
