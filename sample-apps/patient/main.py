import logging
import os
from typing import Dict
import json
import lib.configs

from monailabel.interfaces.app import MONAILabelApp
from monailabel.interfaces.config import TaskConfig
from monailabel.interfaces.tasks.infer_v2 import InferTask 
from monailabel.interfaces.tasks.scoring import ScoringMethod
from monailabel.interfaces.tasks.strategy import Strategy
from monailabel.interfaces.tasks.train import TrainTask
from monailabel.utils.others.planner import HeuristicPlanner
from monailabel.datastore.patient import PatientDatastore
from monailabel.sam2.infer import Sam2InferTask

from monailabel.utils.others.class_utils import get_class_names
from monailabel.utils.others.generic import get_bundle_models, strtobool

logger = logging.getLogger(__name__)

class MyPatientApp(MONAILabelApp):
    def __init__(self, app_dir, studies, conf):
        self.model_dir = os.path.join(app_dir, "model")

        configs = {}
        for c in get_class_names(lib.configs, "TaskConfig"):
            name = c.split(".")[-2].lower()
            configs[name] = c

        configs = {k: v for k, v in sorted(configs.items())}

        # Load models from app model implementation, e.g., --conf models <segmentation_spleen>
        models = conf.get("models")
        if not models:
            print("")
            print("---------------------------------------------------------------------------------------")
            print("Provide --conf models <name>")
            print("Following are the available models.  You can pass comma (,) seperated names to pass multiple")
            print(f"    all, {', '.join(configs.keys())}")
            print("---------------------------------------------------------------------------------------")
            print("")
            exit(-1)

        models = models.split(",") if models else []
        models = [m.strip() for m in models]
        # Can be configured with --conf scribbles false or true
        self.scribbles = conf.get("scribbles", "true") == "true"
        invalid = [m for m in models if m != "all" and not configs.get(m)]
        if invalid:
            print("")
            print("---------------------------------------------------------------------------------------")
            print(f"Invalid Model(s) are provided: {invalid}")
            print("Following are the available models.  You can pass comma (,) seperated names to pass multiple")
            print(f"    all, {', '.join(configs.keys())}")
            print("---------------------------------------------------------------------------------------")
            print("")
            exit(-1)

        # Use Heuristic Planner to determine target spacing and spatial size based on dataset+gpu
        spatial_size = json.loads(conf.get("spatial_size", "[48, 48, 32]"))
        target_spacing = json.loads(conf.get("target_spacing", "[1.0, 1.0, 1.0]"))
        self.heuristic_planner = strtobool(conf.get("heuristic_planner", "false"))
        self.planner = HeuristicPlanner(spatial_size=spatial_size, target_spacing=target_spacing)

        # app models
        self.models: Dict[str, TaskConfig] = {}
        for n in models:
            for k, v in configs.items():
                if self.models.get(k):
                    continue
                if n == k or n == "all":
                    logger.info(f"+++ Adding Model: {k} => {v}")
                    self.models[k] = eval(f"{v}()")
                    self.models[k].init(k, self.model_dir, conf, self.planner)
        logger.info(f"+++ Using Models: {list(self.models.keys())}")

        # Load models from bundle config files, local or released in Model-Zoo, e.g., --conf bundles <spleen_ct_segmentation>
        self.bundles = get_bundle_models(app_dir, conf, conf_key="bundles") if conf.get("bundles") else None

        super().__init__(
            app_dir=app_dir,
            studies=studies,
            conf=conf,
            name=f"MONAILabel - Patient Analysis",
            description="Deep learning models for multi-image patient analysis",
            version="0.1",
        )

    def init_datastore(self):
        logger.info(f"Init PatientDatastore for: {self.studies}")
        datastore = PatientDatastore(
            datastore_path=self.studies,
            extensions=self.conf.get("extensions", ["*.nii.gz", "*.nii"]),
            auto_reload=True,
            read_only=False,
        )
        if self.heuristic_planner:
            self.planner.run(datastore)
        return datastore

    def init_infers(self) -> Dict[str, InferTask]:
        infers: Dict[str, InferTask] = {}

        if self.models:
            for n, task_config in self.models.items():
                c = task_config.infer()
                c = c if isinstance(c, dict) else {n: c}
                for k, v in c.items():
                    logger.info(f"+++ Adding Inferer:: {k} => {v}")
                    infers[k] = v

        # Add SAM support
        sam_model = Sam2InferTask(
            self.model_dir,
            type="segmentation",
            dimension=3,
            description="Segment Anything Model (SAM) for interactive segmentation",
        )
        infers["sam"] = sam_model

        return infers

    def init_trainers(self) -> Dict[str, TrainTask]:
        trainers: Dict[str, TrainTask] = {}
        # Add training tasks if needed
        return trainers

    def init_strategies(self) -> Dict[str, Strategy]:
        strategies: Dict[str, Strategy] = {}
        # Add active learning strategies if needed  
        return strategies

    def init_scoring_methods(self) -> Dict[str, ScoringMethod]:
        methods: Dict[str, ScoringMethod] = {}
        # Add scoring methods if needed
        return methods
