
import logging
import qt
import slicer
from slicer.ScriptedLoadableModule import *
from MONAILabelLib import MONAILabelClient
from monailabel.client.patient_utils import PatientClient

class PatientSelector(ScriptedLoadableModuleWidget):
    def __init__(self, parent=None):
        ScriptedLoadableModuleWidget.__init__(self, parent)
        self.logic = None
        self.client = None
        self.patient_client = None
        self.patients = {}

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        
        # Create layout
        layout = qt.QVBoxLayout(self.parent)
        
        # Server section
        serverCollapsibleButton = ctk.ctkCollapsibleButton()
        serverCollapsibleButton.text = "Server"
        layout.addWidget(serverCollapsibleButton)
        
        serverFormLayout = qt.QFormLayout(serverCollapsibleButton)
        
        # Server address
        self.serverComboBox = qt.QComboBox()
        self.serverComboBox.setEditable(True)
        self.serverComboBox.setSizePolicy(qt.QSizePolicy.Expanding, qt.QSizePolicy.Fixed)
        serverFormLayout.addRow("Server Address:", self.serverComboBox)
        
        self.connectButton = qt.QPushButton("Connect")
        serverFormLayout.addRow("", self.connectButton)
        
        # Patient section
        patientCollapsibleButton = ctk.ctkCollapsibleButton()
        patientCollapsibleButton.text = "Patients"
        layout.addWidget(patientCollapsibleButton)
        
        patientFormLayout = qt.QFormLayout(patientCollapsibleButton)
        
        self.patientSelector = qt.QComboBox()
        patientFormLayout.addRow("Patient:", self.patientSelector)
        
        # Images section  
        imagesCollapsibleButton = ctk.ctkCollapsibleButton()
        imagesCollapsibleButton.text = "Patient Images"
        layout.addWidget(imagesCollapsibleButton)
        
        imagesFormLayout = qt.QFormLayout(imagesCollapsibleButton)
        
        self.imagesList = qt.QListWidget()
        self.imagesList.setSelectionMode(qt.QAbstractItemView.ExtendedSelection)
        imagesFormLayout.addRow("Images:", self.imagesList)
        
        self.loadImagesButton = qt.QPushButton("Load Selected")
        imagesFormLayout.addRow("", self.loadImagesButton)
        
        # Connect signals
        self.connectButton.connect("clicked(bool)", self.onConnectButton)
        self.patientSelector.connect("currentIndexChanged(int)", self.onPatientSelected)
        self.loadImagesButton.connect("clicked(bool)", self.onLoadImagesButton)
        
        # Add vertical spacer
        self.layout.addStretch(1)

    def updateServerUrlGUIFromSettings(self):
        settings = qt.QSettings()
        serverUrlHistory = settings.value("MONAILabel/serverUrlHistory")
        
        self.serverComboBox.clear()
        if serverUrlHistory:
            self.serverComboBox.addItems(serverUrlHistory.split(";"))
        self.serverComboBox.setCurrentText(settings.value("MONAILabel/serverUrl"))

    def onConnectButton(self):
        server_url = self.serverComboBox.currentText.strip()
        if not server_url:
            slicer.util.errorDisplay("Please enter a server URL")
            return
        
        # Save server URL
        settings = qt.QSettings()
        settings.setValue("MONAILabel/serverUrl", server_url)
        
        # Save URL history
        serverUrlHistory = settings.value("MONAILabel/serverUrlHistory")
        if serverUrlHistory:
            serverUrlHistory = serverUrlHistory.split(";")
        else:
            serverUrlHistory = []
        try:
            serverUrlHistory.remove(server_url)
        except ValueError:
            pass
        serverUrlHistory.insert(0, server_url)
        serverUrlHistory = serverUrlHistory[:10]  # keep up to first 10 elements
        settings.setValue("MONAILabel/serverUrlHistory", ";".join(serverUrlHistory))
        
        try:
            # Connect to server
            qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
            self.client = MONAILabelClient(server_url, "user-xyz")
            self.patient_client = PatientClient(self.client)
            
            # Get patient list
            self.patients = self.patient_client.get_all_patients()
            
            # Update patient selector
            self.patientSelector.clear()
            for patient_id in self.patients:
                self.patientSelector.addItem(patient_id)
                
            slicer.util.infoDisplay(f"Connected to server. Found {len(self.patients)} patients.")
        except Exception as e:
            slicer.util.errorDisplay(f"Failed to connect to server: {str(e)}")
        finally:
            qt.QApplication.restoreOverrideCursor()

    def onPatientSelected(self):
        self.imagesList.clear()
        patient_id = self.patientSelector.currentText
        if not patient_id:
            return
        
        # Get images for selected patient
        if patient_id in self.patients:
            for image_id in self.patients[patient_id]:
                self.imagesList.addItem(image_id)

    def onLoadImagesButton(self):
        selected_items = self.imagesList.selectedItems()
        if not selected_items:
            slicer.util.errorDisplay("No images selected")
            return
        
        try:
            qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
            for item in selected_items:
                image_id = item.text()
                
                # Download image
                image_uri = self.client.download_image(image_id)
                
                # Load into Slicer
                volume_node = slicer.util.loadVolume(image_uri)
                if volume_node:
                    volume_node.SetName(image_id)
                else:
                    slicer.util.errorDisplay(f"Failed to load {image_id}")
                    
            # Switch to appropriate layout based on number of volumes loaded
            if len(selected_items) > 1:
                slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutOneByThreeSliceView)
            else:
                slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutDefaultView)
                
            slicer.util.resetSliceViews()
        except Exception as e:
            slicer.util.errorDisplay(f"Failed to load images: {str(e)}")
        finally:
            qt.QApplication.restoreOverrideCursor()
