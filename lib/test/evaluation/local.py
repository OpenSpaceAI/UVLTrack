from lib.test.evaluation.environment import EnvSettings
import os
prj_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

def local_env_settings():
    settings = EnvSettings()
    settings.prj_dir = prj_dir
    settings.save_dir = prj_dir
    settings.result_plot_path = os.path.join(prj_dir, 'test/result_plots')
    settings.results_path = os.path.join(prj_dir, 'test/tracking_results')
    settings.lasot_path = os.path.join(prj_dir, 'data/lasot')
    settings.nfs_path = os.path.join(prj_dir, 'data/nfs')
    settings.otb_path = os.path.join(prj_dir, 'data/otb99')
    settings.trackingnet_path = os.path.join(prj_dir, 'data/trackingnet')
    settings.uav_path = os.path.join(prj_dir, 'data/uav')
    settings.tnl2k_path = os.path.join(prj_dir, 'data/tnl2k/test')
    settings.otb99_path = os.path.join(prj_dir, 'data/otb99')
    settings.lasot_ext_path = os.path.join(prj_dir, 'data/lasotext')

    return settings