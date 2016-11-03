
import phenix.refinement.command_line
from phenix.refinement import runtime
import libtbx.phil
from libtbx.phil import tokenizer
from libtbx.phil import find_scope
from cStringIO import StringIO
import copy

den_phil = libtbx.phil.parse("""
den {
    include scope mmtbx.den.den_params
}""", process_includes=True)

den_phil_str_overrides = """
refinement {
  main {
    number_of_macro_cycles = 1
  }
  refine {
    strategy = *individual_sites individual_sites_real_space rigid_body \
               *individual_adp group_adp tls occupancies group_anomalous \
               *den
  }
  tardy {
    mode = *every_macro_cycle second_and_before_last once first first_half
    start_temperature_kelvin = 3300
    final_temperature_kelvin = 300
    time_step_pico_seconds = 0.001
    number_of_cooling_steps = 1440
    prolsq_repulsion_function_changes {
      c_rep = 100
      k_rep = 1.0
    }
  }
  simulated_annealing {
    start_temperature = 3300
    final_temperature = 300
    cool_rate = 50
    number_of_steps = 24
    time_step = 0.001
  }
}
"""


def validate_params_den (params) :
  from libtbx.utils import Sorry
  from phenix.refinement import runtime
  basic_result = runtime.validate_params_basic(params=params,
    ignore_generate_r_free_conflict=True)
  if (params.refinement.input.neutron_data.file_name is not None) :
    raise Sorry("Neutron data is not currently compatible with DEN refinement.")
  if (params.refinement.pdb_interpretation.peptide_link.ramachandran_restraints) :
    raise Sorry("Ramachandran restraints are not compatible with DEN refinement.")
  if (params.refinement.main.simulated_annealing):
    raise Sorry("Cartesian simulated annealing should not be turned on for "+
      "DEN refinement. Annealing settings are controlled in the den "+
      "parameter scope.")
  elif (params.refinement.main.simulated_annealing_torsion):
    raise Sorry("Torsion simulated annealing should not be turned on for "+
      "DEN refinement. Annealing settings are controlled in the den "+
      "parameter scope.")
  if (params.refinement.den.final_refinement_cycle and \
      "individual_sites" not in params.refinement.refine.strategy):
    raise Sorry("A final round of refinement requires individual_sites "+
      "to be active.")
  if ("cartesian" in params.refinement.den.annealing_type and \
      "individual_sites" not in params.refinement.refine.strategy):
    raise Sorry("Cartesian DEN refinement requires individual_sites "+
      "to be active.")
  if ("individual_sites_real_space" in params.refinement.refine.strategy):
    raise Sorry("Real space refinement is not recommended for use with "+
      "DEN refinement.")
  #if (params.refinement.main.number_of_macro_cycles != 1) :
  #  raise Sorry("Only 1 macrocycle of refinement is necessary for DEN "+
  #    "refinement.")
  return True

# run by GUI
def validate_params (params) :
  validate_params_den(params)
  runtime.validate_params_gui(params)
  return True

_master_params = None
def master_params () :
  global _master_params
  if (_master_params is None) :
    import iotbx.phil
    master_phil = copy.deepcopy(phenix.refinement.master_params())
    strategy_def = find_scope(master_phil, "refinement.refine.strategy")
    strategy_def.words.append(tokenizer.word("*den"))
    strategy_def.caption += " DEN"
    top_scope = find_scope(master_phil, "refinement")
    top_scope.adopt(den_phil.objects[0])
    # FIXME why is this necessary???
    phil_out = StringIO()
    master_phil.show(out=phil_out, attributes_level=3)
    master_phil = iotbx.phil.parse(phil_out.getvalue())
    #print top_scope.objects[-1].full_path()
    #print top_scope.objects[-1].objects[0].full_path()
    phil_objects = [
      libtbx.phil.parse(input_string=den_phil_str_overrides)]
    _master_params = master_phil.fetch(sources=phil_objects)
  return _master_params

def run(command_name, args) :
  master_phil = master_params()
  phenix.refinement.command_line.run(
    command_name=command_name,
    args=args,
    customized_master_params=master_phil)

class run_den_refine (runtime.run_phenix_refine) :
  allow_add_hydrogens = False
  def get_master_phil (self) :
    return master_params()

  def _run (self, args, callback) :
    master_phil = master_params()
    return phenix.refinement.command_line.run(
      command_name="phenix.den_refine",
      args=args,
      call_back_handler=callback,
      customized_master_params=master_phil)
