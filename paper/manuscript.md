# Local language model agents for monitoring and supervising fused filament 3D printing

Ahmed Ben Sidhom, Aymen Bahri, Essa Al Amiri, Antonios Kontsos

## Abstract

Printer operators need to know what is happening during a print and what action to take. We present a local artificial intelligence (AI) agent system for fused filament fabrication, a form of 3D printing. A supervisor sends data questions to a database agent and printer requests to a printing agent. Tools return records or confirm that a request was accepted. Separate camera and contact acoustic emission (AE) services can request a pause without waiting for the AI conversation. The system keeps agents, tools, stored data and computing tasks separate. An AMD Ryzen AI Max+ 395 computer with 64 GB memory is the designated language-model deployment host. NVIDIA hardware supports visual-model training. We describe labeling, training, storage and tool calls, and an architecture linking records by job, time and commanded position. Supporting tests compare six visual models and five AE classifiers, each fitted with three random seeds on fixed data splits. YOLO11n detection reaches image macro F1 of 0.940 ± 0.043, which combines precision and recall with equal class weighting. Its box average precision at an intersection-over-union threshold of 0.50 (AP50) is 0.720 ± 0.013. A small AE multilayer perceptron reaches macro F1 of 0.773 ± 0.000. These scores support recognition-model selection. They do not measure agent-task success or paired-sensor fusion performance.

Keywords: fused filament fabrication; language model agents; acoustic emission; computer vision; multimodal monitoring; supervisory control

## 1 Introduction

Fused filament fabrication (FFF) makes a part by placing heated thermoplastic along a planned path. The printer adds one layer at a time. Good printing depends on steady plastic flow, suitable temperature, correct movement and good adhesion. A problem can grow over several layers and waste material before the operator notices it. Monitoring should detect a change, connect it to the current print and help the operator choose a response. Reviews of material-extrusion monitoring discuss these linked tasks (Fu et al., 2021; Lu & Wong, 2018; Oleff et al., 2021).

Each sensor gives a different view of the process. Color images show the part's shape and surface. They can reveal gaps or tangled plastic. Contact acoustic emission (AE) records short bursts of elastic waves that travel through the material and mounting path to the sensor. Printer readings give temperature and machine-state information. These sources can help explain each other. To use them together, the system needs clear record types, time rules and decision rules. It must also move data correctly between sensors, storage, displays and supervision.

Large language models (LLMs) let operators ask for records and printer actions in everyday language. Manufacturing systems use specialized agents and tools to connect these requests to process data (Elbasheer et al., 2026; Holland & Chaudhari, 2024; Lim et al., 2024). Our agents work alongside the sensor services. A rule based on consecutive detections can request a pause without waiting for an LLM answer. This keeps the sensor response separate from the conversation. We can then evaluate sensing, reasoning and operator interaction separately.

The best visual model depends on the output we need. An image classifier gives one label for the whole image. An object detector also predicts bounding boxes around possible defects. Boxes help an operator see where to look, but they require more detailed labels for training. A whole-image label may be enough for an alarm. Classification accuracy and box average precision measure different things, so we should not compare them as if they were the same score. We use the same images for both model types. We compare their image decisions and check box quality separately.

AE model choice also depends on accuracy and computing cost. Linear models, groups of decision trees and small neural networks can all use features taken from an acoustic burst. Their decision thresholds, input scaling and speed affect the monitoring service. A decision threshold is the score needed to choose a class. A model may rank events well but give weaker normal-or-defective decisions at one threshold. We therefore check more than overall accuracy. We also check performance across classes, missed defects, false alarms and the quality of probability estimates.

We study how an operator's request reaches the right data source or printer service. We also study how the system returns an answer that the operator can check. We describe the supervisor, its specialist agents and the information passed through each tool. We test the visual and acoustic models that provide this information. The recognition models supply evidence to the agents, so the components and agent tasks require separate evaluation.

This paper has three main contributions. We define a workflow with separate database and printing agents, clear tool inputs and outputs, and an independent sensor pause path. We describe how to link images, AE records and printer readings by job and time. We keep commanded positions separate from measured wave-source locations. We also compare six visual models and five AE models on fixed data splits. These tests give a repeatable basis for choosing the sensing models. Fusion means linking sensor records in the system design; we do not report a measured gain from combining them. Sections 3 and 4 describe the system and recognition tests. Section 5 gives the recognition results. Section 6 explains what is still needed to test complete agent tasks and printing outcomes.

## 2 Related work

### 2.1 Language interfaces and evaluation

Manufacturing researchers use LLMs for planning, agent coordination and natural-language operator requests (Elbasheer et al., 2026; Holland & Chaudhari, 2024; Lim et al., 2024; Zhao et al., 2026). Jadhav et al. (2025) connect language models to 3D-printer monitoring and control. Vyas and Mercangöz (2025) and Schall (2026) separate suggested actions from the checks and software that carry them out. This matters for printer supervision. Reading a record and changing a machine setting have different effects.

These studies support an interface in which we can follow each request, tool call and returned record. Our design shows this path for database searches and OctoPrint operations. The conversation is separate from background sensing, which can request a pause on its own. We can test record selection, command inputs, recognition and physical printer response separately. A good written answer alone does not prove that every stage worked.

Reviews of machine learning in additive manufacturing discuss data quality and deployment as well as model training (Kumar et al., 2023; Qin et al., 2022). Closely related samples or choices based on test results can make a model seem better on new data than it really is (Kapoor & Narayanan, 2023). Probability quality is a separate issue. A high model score is not always a reliable probability estimate (Guo et al., 2017). We keep the data splits fixed, choose decision thresholds using validation data and state what each speed test includes. We describe sensor linking as a system design. Our numerical tests cover the individual recognition models.

### 2.2 Camera monitoring and print correction

Camera monitoring has moved from direct image checks to learned defect recognition and corrective actions. Straub (2015) studied software image analysis for additive manufacturing. Holzmond and Li (2017) developed defect detection during printing. Petsiuk and Pearce (2020) presented an open-source method for checking individual layers. Charalampous et al. (2021) studied real-time printing error detection. Banadaki et al. (2022) used deep convolutional networks for automated quality and process control. These studies examine different outputs, including layer differences, defect labels and defect locations.

To move from finding a defect to fixing it, a test must include the printing action. Brion and Pattinson (2022) linked images to controlled changes in printer settings and tested correction under different conditions. Fu et al. (2023) connected process observations to checks of part structure. Yu et al. (2025) used several views and combined information for monitoring and automatic correction. These studies show why we should check both recognition and its effect on the part. Scores depend on the defect being studied, the sensor setup and what counts as one test sample.

Other optical tools can measure more than a ordinary color image. Lin et al. (2019) used laser scanning to measure deposited shape. Sadaf and Ahmed (2026) studied thermal image sequences and printer-setting changes. These methods add direct shape or temperature measurements. Our system uses a webcam for visible appearance and printer readings for temperature context. We therefore choose visual models for that sensor setup.

### 2.3 Acoustic emission and other signals

Contact AE can reveal changes through elastic waves carried by the printer and material. Yang et al. (2018) used AE to study filament breakage. Liu et al. (2018) studied AE-based fault diagnosis in fused deposition modeling. Li et al. (2022) monitored polymer flow with contact AE. Xu et al. (2022) combined AE and laser scanning. These studies support using acoustic records alongside images. The two sensors observe the process in different physical ways.

An AE feature depends on the sensor, its contact with the machine, signal gain, filtering and the rules used to separate events. Values such as amplitude, energy and counts need clear measurement settings. Forster et al. (2026) discuss ways to inspect material extrusion without damaging the part. Rooney et al. (2025) use feeder-motor current as another process signal. Waheed et al. (2025) study sound traveling through air. Airborne sound and contact AE follow different paths. Their features and scores cannot be treated as the same measurements.

We give every acoustic model the same six calculated features. This lets us compare model types and decision cutoffs without changing how the input features are made. The comparison adds to waveform-based studies by showing which small models work well with the available event-feature data.

### 2.4 Combined monitoring and digital models

Digital-twin research studies how a physical process connects to its digital model and exchanges information with it (Jones et al., 2020; Kritzinger et al., 2018; Tao et al., 2019). L. Chen et al. (2023) link several sensors to a digital-twin framework. Großeheide et al. (2025) use a voxel-based data model, built from small 3D cells, to link different measurements. Rachmawati et al. (2023) study printer fault detection with a digital twin. These works support clear links between records, time, geometry and print jobs.

Our system uses saved records, printer state and a reconstructed toolpath to help the operator. A job identifier groups related records. Timestamps show the relevant time interval. Separate image and acoustic records keep their own meaning. Positions taken from G-code describe where the printer was told to move. This digital view helps the agents link information. It is not a calibrated measurement of actual movement or a tested model that predicts part quality.

## 3 System design and deployment

### 3.1 Hardware and computing tasks

The system connects a Creality Ender 3 V2 printer, Logitech C270 camera and contact AE sensor to a Raspberry Pi and a local computer. Figure 1(a) shows the sensor fixture near the extrusion area. Figure 1(b) shows the camera and sensor connections, including the sensor cable to spotWave. The Raspberry Pi 4 supports OctoPrint, printer communication and remote data access. Figure 1(c) separates the designated AMD deployment host from the NVIDIA visual-training host. Printer access, AI supervision and offline training have different computing roles.

The deployment computer has an AMD Ryzen AI Max+ 395, 64 GB of shared memory and integrated Radeon 8060S graphics. The processor has 16 central processing unit (CPU) cores and 32 threads. The graphics hardware has 40 compute units (AMD, n.d.). Software and graphics can use the shared memory, but model weights, working buffers, conversation context and other applications all take space. The AMD computer is the designated local LLM deployment host. The 4 GB NVIDIA RTX 3050 Laptop graphics processing unit (GPU) is used for the recognition tests. Their performance figures describe different machines and tasks.

Table 1 Hardware and assigned tasks

| Component | Capability or interface | Assigned workload |
| --- | --- | --- |
| AMD Ryzen AI Max+ 395 | 16 CPU cores and 32 threads; Radeon 8060S; 64 GB installed unified memory | Local LLM service and supervisory application; distinct from the training GPU |
| NVIDIA RTX 3050 Laptop GPU | 4 GB dedicated memory in the benchmark host | Visual-model training and recognition timing; not LLM deployment |
| Raspberry Pi 4 | USB and Ethernet or wireless networking | OctoPrint, printer communication and remote acquisition access |
| Ender 3 V2 | FFF printer with OctoPrint interface | Physical deposition and controller operations |
| Logitech C270 | 720p at 30 fps product rating | Camera stream; processed-frame interval includes the service loop |
| VS600-Z1 | Passive contact sensor; nominal 600 kHz resonance; specified 550–730 kHz band | High-frequency structure-borne acoustic transients |
| spotWave 201 | One channel; 16-bit conversion; specified 2 MHz sampling | AE acquisition and event records |

Hardware specifications describe what each device can support. The AMD computer has 64 GB memory. NVIDIA benchmark times are not AMD language-model times.

![Figure 1](../assets/figures/figure_01.png)

Fig. 1 Hardware setup

The camera sends images through an HTTP video connection. Its rated output is 720p at 30 frames per second. This is the camera's stream capacity. The monitoring service waits one second after each processing loop (Logitech, n.d.). The Raspberry Pi provides USB and network connections (Raspberry Pi Ltd, n.d.). AE records are sent as structured messages over SSH, a secure remote connection. OctoPrint provides printer state and operations through its REST application programming interface (API), a software request-and-reply interface. The Pi's data-access work remains separate from model serving and training.

The Vallen VS600-Z1 is a passive contact AE sensor. Its nominal resonance is near 600 kHz, and its stated frequency band is 550–730 kHz. The spotWave 201 has one data channel, 16-bit conversion and a rated sampling rate of 2 MHz (Vallen Systeme, n.d.-a, n.d.-b). The fixture carries elastic waves from near the extrusion area to the sensor. This records short, high-frequency waves through solid material. It does not measure airborne sound or the low-frequency vibration measured by an accelerometer. Mounting, contact, gain and filtering affect how the recorded signal relates to the printing process.

### 3.2 Agents and the local language model

The workflow has three agents with different jobs. The supervisor reads the operator's request and decides which information or action is needed. The database agent gets process records. The printing agent reads printer state and requests supported actions. Figure 2 separates these agents from the tools they call and the resources they use. SQLite stores records. OctoPrint connects to the printer. Ollama runs the shared language-model service. Background sensing and pause monitoring work alongside the conversation. They are not extra agents. The logos identify software and hardware, not separate reasoning roles.

Table 2 Agent tasks and step limits

| Agent role | Responsibility and access | Execution configuration |
| --- | --- | --- |
| Supervisory agent | Interprets requests; delegates to database and printing roles; combines returned evidence | 20 iterations inherited from smolagents 1.22.0; no direct application tool |
| Database agent | Selects and retrieves records through the database query tool | 2 iterations; tabular, numerical and image-handling libraries |
| Printing agent | Reads state and requests operations through the printer service tool | 3 iterations; HTTP, timing and structured-message libraries |

The agents share one local model service and run code through local Python. Their step limits do not set a printer-response deadline. They do not provide separate operating-system sandboxes.

We choose Qwen3.6-35B-A3B through Ollama for the upgraded deployment. It has about 35 billion model parameters, with about 3 billion active for each token, or small piece of text. Its model card reports better results than Qwen3.5-35B-A3B on several coding and tool tests, but not on every task (Qwen Team, 2026). Ollama offers a package using the Q4_K_M quantization format of about 24 GB, including related components (Ollama, n.d.-b). This is a practical candidate for the 64 GB AMD computer. The package size is not the total memory needed while running. We use the model for text and code tasks. We do not claim that the LLM directly interprets AE waveforms.

![Figure 2](../assets/figures/figure_02.png)

Fig. 2 Agent workflow

The source implementation uses Qwen3-Coder 30B with an 8,192-token application context (Ollama, n.d.-c). We select Qwen3.6 as a documented deployment upgrade. The upgrade has not yet been evaluated on manufacturing-agent tasks. Acceptance tests must check record retrieval, tool inputs, error handling, memory use and speed on the AMD computer. Ollama lists support for the Ryzen AI Max+ 395 with Linux ROCm (Ollama, n.d.-a). The drivers and software must be compatible. We do not assign the LLM to the neural processing unit (NPU) or assume acceleration from the device name.

Table 2 gives the agents' step limits. The database agent allows two iterations, and the printing agent allows three. The supervisor uses the default of 20 in smolagents 1.22.0 (Hugging Face, 2025). An iteration is one pass through the reasoning and execution loop. It can include several function calls. These limits do not state how quickly the physical printer will respond. The database role uses table and image libraries. The printing role uses time, HTTP and structured-message libraries. These jobs remain the same when we change the language-model service.

### 3.3 Tools and checking records before acting

Two application tools connect the agents to data and printer services. The database tool receives an SQL statement, which is a request for database records. It checks SQL syntax by parsing the statement, then uses a table library to execute the query and return SQLite rows. This syntax check is not an authorization policy. It returns a table, a message that no rows were found, or an error. The interface shows the query and its result. The printer tool receives an operation and any needed filename, target temperature, feed setting or flow setting. It then calls OctoPrint. Table 3 lists these inputs, replies and effects. Camera capture, feature processing and saving records run in the background. They are not conversation tool calls.

Table 3 Tool inputs, outputs and effects

| Tool | Input and operation | Returned information and effects |
| --- | --- | --- |
| Database query | SQL statement; parse and query the SQLite observation tables | Rows, empty result or error; displays the executed query and returned table |
| Printer service | Operation with optional filename, temperature, flow or feed argument; call OctoPrint | Files, formatted state, acknowledgment or error; write operations can change the printing process |

Tools are software interfaces that an agent can call. They are not agents or models. Production use still needs separate action checks and code isolation.

Figure 3 follows a request through the supervisor, specialist, tool and resource. Solid arrows show requests and calls. Dashed arrows show returned records or replies. To summarize recent detections, the supervisor asks the database agent for the records. That agent writes the query. The tool gets matching rows and returns them. The specialist passes the result to the supervisor, which uses the records to answer. If the request also includes a printer action, the supervisor must call the printing agent separately. The figure shows the intended call order. It is not a measured task trace, and arrow spacing does not show speed.

Supported printer operations include listing files, starting, pausing, resuming or canceling a job, reading state, setting temperature targets and changing feed or flow percentages. Their effects differ. Reading state only retrieves information. A job or temperature command can change the process. The system must keep three things separate: the requested action, the API reply and the printer state observed afterward. A successful HTTP reply does not prove that motion stopped or the target temperature was reached (OctoPrint Project, n.d.).

![Figure 3](../assets/figures/figure_03.png)

Fig. 3 Tool calls

Prompts guide which agent receives a task. Separate software should decide whether a machine action is allowed. Before an action, a production gateway should check the current job, record age, input limits and required approval. The source implementation has application wrappers, but no separately tested policy gateway. A plausible answer or valid SQL statement does not prove that a printer action is authorized.

### 3.4 Code execution and isolation

The framework runs generated code in its local Python interpreter. Figure 3 separates agent reasoning, the executor that calls a tool, and the outside resource that carries out the operation. Import restrictions limit which libraries code can use. Step limits restrict the loop. Neither creates an isolated operating-system sandbox. Ollama serves model output. It does not isolate Python code or approve printer commands. These limits matter because code errors can affect the host, while action checks belong in the tool service.

The production design limits generated code's access to files, network destinations, memory, CPU usage and execution time. A separate tool gateway controls database permissions and printer actions. Both are needed. A container can isolate code that still asks for an unsafe temperature. A printer wrapper can restrict actions while its host code remains poorly isolated. These controls need integration tests with the chosen agent framework. In the pinned version, managed agents cannot simply be moved to a remote executor without further integration work (Hugging Face, n.d., 2025).

### 3.5 Detection services and saved records

The visual service returns defect regions. Each region has a class, confidence score and bounding box in the image. One image can produce several rows linked to the same saved file. The reference service uses a YOLOv5-based checkpoint. Section 5 selects YOLO11n for the upgrade because it gives both an image decision and defect locations. The AE service gives event decisions and signal features, not image boxes. The database must keep these different record types separate, even when the operator checks the same time interval.

The acoustic service receives event features and waveforms. It uses six features: amplitude, duration, energy, root mean square amplitude, rise time and counts. A classifier labels the event, and the waveform remains available to view. We select the small multilayer perceptron (MLP), a neural network with connected layers, tested in Section 5 for the upgrade. The source application's Keras checkpoint and the benchmark MLP are different model files. To transfer the selected model, we must keep its feature order, input processing, label meaning and decision threshold.

SQLite stores visual detections and acoustic events. Image files hold the pictures, and acoustic records can hold saved waveform data. In-memory buffers also keep recent images, temperatures, positions and AE records for display. Query results are kept separately in the interface session. A visual row describes one detection, not always a whole image. An AE feature row is not an image. An image with no detection may produce no detection row. A missing row therefore does not prove that printing was normal.

Figure 4 shows the planned links between print jobs, images and acoustic events. A job identifier connects related records. Each sensor's records keep their event time, class and confidence. Acoustic records also keep features and waveform data. We use Mermaid to draw the colored database diagram (Mermaid, n.d.). Crow's foot notation shows that one job can have many records. The design helps the database agent search within a job. The observation tables exist. The diagram alone does not prove that SQLite enforces all job links through foreign-key checks.

![Figure 4](../assets/figures/figure_04.png)

Fig. 4 Data links

The current saving process adds a timestamp from the host computer. Linking sensor streams also needs acquisition time, receipt time, a record identifier and a model identifier. Two records saved close together were not necessarily measured at the same time. A lasting interaction log must store the operator's request, delegated calls, returned records and action results. Keeping only the latest displayed query is not enough.

The monitoring services can request a pause directly. Three consecutive positive visual frames or five consecutive defective acoustic events trigger a pause request. A processed negative observation resets that sensor's count. These rules run without the LLM conversation. Their behavior depends on sampling, missing records and processing delay. Recognition accuracy and actual printer-interruption performance must therefore be measured separately.

### 3.6 Visual defect regions and acoustic event association

Figure 5(a) shows a defect region from the retained detector example. It is not a new YOLO11n prediction. Panel (b) shows the contact AE fixture and event-classification interface. Panel (c) rebuilds the rectangular specimen from real G-code at layer 64 of 127, where Z = 12.8 mm. It includes 11,060 earlier extrusion segments and 225 segments in the current layer. These include walls, skin and infill. The path shows commanded deposition, not measured nozzle movement.

![Figure 5](../assets/figures/figure_05.png)

Fig. 5 Sensing and toolpath association

The three small waveform plots in Figure 5(c) use different recorded AE events. We convert voltage from volts to millivolts using the Vallen WaveLine interface values. We do not normalize their amplitudes (Vallen Systeme, n.d.-c). Time is shown in microseconds relative to the trigger. We infer a 1 µs sample interval from all 35 saved records. Their sample count equals the integer part of duration in seconds multiplied by 10⁶, plus 200 pre-trigger samples. This inferred interval is different from spotWave's rated 2 MHz capacity. The recorded waveforms and G-code are recorded data. The AE1–AE3 position links are schematic because the saved events have no XYZ coordinates.

For camera detection, we measure location quality by comparing predicted boxes with labeled boxes. For AE, the system links an event to commanded coordinates. It does not find the wave source through triangulation. One contact sensor and commanded positions cannot independently locate where a wave began inside the material. A marker can help the operator inspect the related printing interval. A measured source-location claim would need known physical positions and measured position error. We compare the information each sensor provides. Their different scores are not interchangeable measures of spatial accuracy.

The fusion design links records by a shared job identifier and a time window. It keeps image regions, acoustic decisions, waveform links and printer context. The association design must retain missing and stale-record status rather than replace absent observations with unrelated data. These links give the supervisor more context. We do not report a numerical gain from fusion. Testing that gain would need synchronized sensor measurements and shared event labels from the same print jobs.

### 3.7 Model choice and deployment requirements

Table 4 gives each model's job and the evidence for its selection. We choose YOLO11n when the operator needs both a defect decision and its image location. We choose the small MLP for the six-feature AE task using macro F1. Image classifiers remain useful baselines because they only need to label the whole frame. The LLM has a different job. It interprets requests and prepares tool instructions. Its acceptance tests must check correct retrieval, valid inputs, useful answers and AMD resource use. Detector AP and acoustic F1 do not measure these tasks.

Table 4 Selected models and supporting evidence

| Role | Selected model or reference | Evidence and integration scope |
| --- | --- | --- |
| Local language model | Qwen3.6-35B-A3B through Ollama on AMD | Selected upgrade; quantized package approximately 24 GB; local tool-task acceptance not yet measured |
| Language-model reference | Qwen3-Coder 30B | Source-configured backend with an 8192-token application context |
| Visual detection and localization | YOLO11n detector | Highest mean image macro F1 among tested candidates; bounding-box AP evaluated separately; selected recognition upgrade |
| AE event decision | Small MLP with 32 and 16 hidden units | Highest mean macro F1 among the five six-feature learners; requires the fitted transform and selected threshold |
| Application recognition reference | YOLOv5-based detector and Keras AE model | Original integrated sensing path and source of the retained interface images |
| Comparators | Four image classifiers; YOLO26n detector; four alternative AE classifiers | Same within-modality fixed test membership and three fitting seeds |

Recognition upgrades are evaluated separately from the retained interface examples. Recognition scores and language-model acceptance tests measure different tasks.

A recognition release must keep the fitted model, input processing, class or feature map and selected cutoff together. Visual training stays on the NVIDIA development computer. AMD inference and sensor integration need their own compatibility and speed tests. The LLM upgrade must also record its version, quantization, application context and serving settings. Quantization controls how model weights are stored with reduced numerical precision. Keeping these settings together lets us update components without changing the meaning of their outputs.

## 4 Recognition tests

### 4.1 Data and evaluation splits

Figure 6 separates image labeling, detector training, saved AE events and the acoustic model. In Figure 6(a), Roboflow receives raw color camera images. These input images have no boxes or prediction overlays. A labeler gives each visible defect a class and a box. A review checks the labels before export (Roboflow, n.d.; Ultralytics, n.d.-a). The output has matching image and label files, plus a configuration file with class names and split locations. Each YOLO label row gives a class identifier, box-center coordinates, width and height. The box values are normalized to image size. Class 0 is spaghettification, or tangled extruded plastic. Class 1 is underextrusion, or too little deposited plastic. An image with no labeled defect has an empty label file.

Figures 6(a) and 6(b) use the same unchanged 640 × 480 camera image of a white rectangular specimen. Panel (b) shows a separate target overlay to explain training. It is not a saved defect label, and we do not claim that this frame is in the benchmark. Recorded dataset-preparation settings include auto-orientation and 640 × 640 resizing. They also include saturation changes by ±25% and brightness changes of ±24%, with blur up to 2.5 pixels. These are preparation settings. The benchmark splits and changes applied during training are described separately below.

![Figure 6](../assets/figures/figure_06.png)

Fig. 6 Model workflows

The visual dataset has 1,235 images, 1,658 labeled objects and 192 images with no object labels. The two box classes are underextrusion and spaghettification. For image classification, an image with no boxes gets the background label. If all boxes in an image have one class, that class becomes the image label. The shared comparison only uses frames that fit this single-label rule. Background is therefore a whole-image category, not a third kind of defect box.

We split the visual data in time order within each acquisition sequence and class. Related source frames and their exported versions are grouped before splitting. The groups are then ordered within each sequence and class. About 60% go to training, 20% to validation and 20% to testing. We remove two source groups on each side of a split boundary. This removes 88 images in 40 boundary groups. The final sets contain 728 training, 200 validation and 219 test images. No source group or identical image file appears in more than one set. Table 5 gives the class counts. Grouping also keeps exported versions with their source frames.

Table 5 Class counts in each data split

| Modality / unit | Class | Training | Validation | Test |
| --- | --- | --- | --- | --- |
| Vision / images | Background | 108 | 20 | 28 |
| Vision / images | Underextrusion | 295 | 87 | 93 |
| Vision / images | Spaghettification | 325 | 93 | 98 |
| Vision / images | Total | 728 | 200 | 219 |
| Vision / boxes | Both defect classes | 944 | 345 | 279 |
| AE / feature rows | Normal | 287 | 96 | 96 |
| AE / feature rows | Defective | 217 | 72 | 72 |
| AE / feature rows | Total | 504 | 168 | 168 |

The visual split excludes 88 boundary images from the 1,235-image dataset. The AE split keeps similar class proportions using seed 17. All fitting seeds use the same data splits.

The visual archive has three acquisition sequences. Sequence test5 contains spaghettification. Sequences test2 and test6 contain background and underextrusion. The time-ordered split tests recognition within these recording conditions. Grouping and removing boundary frames reduce overlap between nearby frame families. However, class and sequence remain linked. This limits what the test can show about performance on new recording conditions.

The AE dataset has 840 observations. Each has six features: amplitude, duration, energy, root mean square amplitude, rise time and counts. Root mean square amplitude describes the overall signal level. The labels include 479 normal and 361 defective observations. We remove the saved row-index column and use only the six features as inputs. A stratified split, which keeps similar class proportions, uses seed 17. It gives 504 training rows, 168 validation rows and 168 test rows. The test set has 96 normal and 72 defective rows. No identical feature vector appears in more than one split or has conflicting labels in this collection.

The 840 feature rows and the 35 saved waveforms in Figure 6(c) serve different purposes. The feature rows train and test classifiers. The waveform archive shows recorded event times and signals. One AE observation is not a whole print job. The feature data have no job-group identifiers and no checked row-to-waveform link. The collections do not provide a verified paired set of print jobs. Both data splits stay fixed across fitting seeds. Test labels are used only for final scoring.

Figure 6(c) shows the times and peak amplitudes of all 35 saved hits. Hit A is the first saved event after 400 s. It occurs at 458.95 s. The close-up shows that same event's waveform. The 15 earlier hits are blue. Hit A and the 19 later hits are red. The colors mean before Hit A and from Hit A onward. They do not prove a change from normal printing to a defect. Every saved event in this archive is labeled defective. The timeline alone cannot establish when a defect started. Voltage conversion and inferred waveform timing follow Section 3.6.

### 4.2 Visual models and training

We compare pretrained YOLO11n and YOLO26n detectors with a YOLO11n classifier and three ImageNet-pretrained classifiers: ResNet-18, MobileNetV3-Small and EfficientNet-B0. These baselines use residual learning, compact mobile design and compound scaling (He et al., 2016; Howard et al., 2019; Tan & Le, 2019). YOLO26 is a newer detector family (Ultralytics, n.d.-b). Every model uses the same eligible images and fitting seeds 17, 29 and 43. A seed sets the starting randomness of a run. Training lasts up to 30 epochs, or passes through the training data. Early stopping allows eight epochs without improvement. Classifiers use 224 × 224-pixel inputs, and detectors use 640 × 640-pixel inputs. This compares practical model pipelines on shared images. It does not give every model equal computing cost.

Figure 6(b) shows YOLO11n's backbone, feature-combining stages and detection heads from the official nano configuration (Ultralytics, n.d.-c). These stages extract image features, combine information at different sizes and predict boxes and classes. The training input contains decoded RGB pixels resized and padded to 640 × 640. Box and class targets are stored separately. We do not paint the true boxes into the input image. Training starts from pretrained weights. It saves the best and last model checkpoints, plus loss, precision, recall and AP logs for each epoch. Validation selects the checkpoint. The held-out test set gives the final scores. At inference, the model only needs an unboxed image. It returns predicted box coordinates, class identifiers and confidence scores.

During torchvision classifier training, we change the images to provide varied examples. We use random resized crops with scale 0.80–1.00, horizontal flips, brightness and contrast jitter of 0.15, and saturation jitter of 0.10. We also apply ImageNet normalization. Evaluation uses resizing, a center crop and the same normalization. The AdamW optimizer starts with learning rate 3 × 10⁻⁴ and weight decay 10⁻⁴. Cosine annealing changes the learning rate over the maximum epoch budget. The loss is cross-entropy with label smoothing of 0.1. CUDA training uses automatic mixed precision, which combines numerical precisions during computation.

The batch size is the number of training examples processed together. We use 16 for ResNet-18 and MobileNetV3-Small, 12 for EfficientNet-B0 and eight for YOLO11n classification. YOLO11n detection uses four. Each detector's own configuration is saved with its run. For torchvision models, we keep the checkpoint with the best validation macro F1. Ultralytics runs keep the best checkpoint chosen by the framework. We do not use final test scores to choose models or decision cutoffs.

The visual tests run on an NVIDIA GeForce RTX 3050 Laptop GPU with 4 GB memory. The saved environment includes Python 3.12.14, PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128 and CUDA 12.8. The speed results describe this computer and software setup. They do not include the Raspberry Pi's communication and data-access tasks.

### 4.3 Image decisions and box locations

We turn detector boxes into one image label using a fixed rule. Let $c_k(x)$ be the highest box confidence for defect class k in image x. If $\max_k c_k(x)$ is below cutoff τ, we label the image as background. Otherwise, we choose the class with the highest confidence. The evaluator first uses a confidence cutoff of 0.001 for candidate detections and uses an intersection-over-union suppression threshold of 0.7. This suppresses overlapping detections according to their box overlap. For each seed, we choose the cutoff with the best validation macro F1. Ties are settled by balanced accuracy, then by the smaller cutoff.

This rule lets us compare whole-image decisions while keeping the detector's box output. We score box localization using average precision (AP), which summarizes precision and recall at different confidence cutoffs. Intersection over union (IoU) is the overlapping box area divided by the union of the box areas. AP50 is average precision at an IoU threshold of 0.50. AP50:95 averages IoU thresholds from 0.50 to 0.95 in steps of 0.05. We report each class and the overall mean. A correct image label does not mean that the model found every labeled object or drew every box correctly.

The detector adapter also creates normalized numerical channels for the image decision. The chosen cutoff contributes the background channel. These values follow the label-selection rule. They are not calibrated probabilities of the classes. We therefore do not rank these detector channels and classifier probabilities together for probability calibration.

### 4.4 Acoustic models and decision cutoffs

We compare logistic regression, random forest, XGBoost, CatBoost and a small MLP. XGBoost and CatBoost combine decision trees to learn nonlinear patterns in table data (T. Chen & Guestrin, 2016; Prokhorenkova et al., 2018). Logistic regression uses balanced class weights and up to 3,000 solver iterations. Random forest uses 500 trees, a minimum leaf size of two and balanced subsample class weights. XGBoost uses 350 estimators, depth four, learning rate 0.035, minimum child weight two, subsampling 0.85 and column subsampling 0.90. CatBoost uses 350 iterations, depth five and learning rate 0.035.

Figure 6(d) shows the fitted MLP as one left-to-right sequence. We export it to Open Neural Network Exchange (ONNX), a model file format, and view it in Netron (Roeder, n.d.). Its six inputs are ordered as amplitude, duration, energy, root mean square amplitude, rise time and counts. The fitted transforms first center and scale these inputs. Two hidden layers then use 32 and 16 units with ReLU activations. ReLU sets negative values to zero. A single sigmoid output gives a defect probability between zero and one. A cutoff chosen on validation data turns this probability into a class. The exported model package keeps the fitted model and transforms together. This diagram shows the model structure. It does not prove physical localization or fusion performance.

The MLP uses regularization α = 0.001, batch size 32 and initial learning rate 0.001. Regularization discourages overly complex fitting. Training allows up to 800 iterations. It stops early after 35 iterations without improvement. An internal validation fraction of 0.15 comes from the 504 training rows. Logistic regression and the MLP fit their standardization during training. Evaluation keeps the same feature order and fitted transform.

We fit every acoustic model with seeds 17, 29 and 43. The separate validation set chooses the binary decision cutoff from 181 values. They run from 0.05 to 0.95 in steps of 0.005. We choose the best macro F1, then balanced accuracy, then the smaller cutoff. We fix that cutoff before testing. This lets each model balance missed defects and false alarms. We do not force every model to use a probability cutoff of 0.5.

### 4.5 Scores and processing time

Accuracy is the share of observations given the correct label. Precision is the share of a class's predictions that are correct. Recall is the share of one true class that the model finds. Balanced accuracy is the mean recall across classes. F1 combines precision and recall: $F1_k = \frac{2TP_k}{2TP_k + FP_k + FN_k}$. Here k identifies the class. TP means true positives, FP means false positives and FN means false negatives. Macro F1 is the mean of the class F1 scores, with equal weight for each class. The visual comparison has three image categories. The acoustic comparison has two classes.

For AE, sensitivity is the share of defective observations correctly found. Specificity is the share of normal observations correctly classified as normal. Area under the receiver operating characteristic curve (AUROC) summarizes how well the scores rank defective and normal observations across decision thresholds. The receiver operating characteristic (ROC) curve plots sensitivity against the false-positive rate. The Brier score averages the squared difference between predicted defect probability and the actual binary label. Expected calibration error (ECE) checks how closely predicted probabilities match observed outcomes. We use 15 equal-width probability bins and weight each bin by its observation count (Guo et al., 2017). We report these checks without adding a separate probability-calibration step after fitting.

We report the mean ± sample standard deviation across three fitting seeds. Standard deviation shows how much the fitting runs vary on the same data splits. It does not measure uncertainty across independent print jobs. A displayed ±0.000 means that variation is smaller than the shown precision. Overlapping summaries are not a formal test that two models are equal or that one is better.

We time visual prediction with a batch size of one after an initial model call. Where needed, CUDA synchronization makes the timing wait for GPU work to finish. The measured path includes file reading, input processing, prediction and output processing. AE timing starts from six features that have already been calculated. We make 20 warm-up calls, then use the first 100 test rows. These measurements exclude physical sensing, acoustic feature calculation, communication and printer response. Saved-checkpoint size separately measures visual-model storage.

### 4.6 Agent evaluation protocol

Agent testing must check the full path in Figures 2 and 3. Recognition F1 cannot stand in for correct agent work. We define tasks for retrieving one job's records, handling empty results, reading printer state, requesting supported actions and recovering from bad inputs or service errors. Each task needs an expected set of records or action, allowed tool inputs and an expected answer based on the returned evidence. Actions that affect the printer should first be tested with a controlled printer interface. An API reply and a physically checked outcome remain different results.

The comparison should use the same tasks and tool rules for three systems: multiple agents, one agent with tools, and a fixed-rule interface without an LLM. Each trace must save the request, delegation, tool inputs, executed query, returned records, error handling and final answer. We should measure exact record-set correctness, valid calls, full task success, unsupported claims and recovery after errors. On the AMD computer, we also need inference time and peak memory. The report must record model version, quantization, context limit and serving settings. These are planned test measures. Section 5 reports recognition results, not results from this agent-task protocol.

## 5 Recognition results

### 5.1 Visual labels and defect boxes

Table 6 compares six models on the same 219 test images. YOLO11n detection has the highest mean macro F1, 0.940 ± 0.043. EfficientNet-B0 follows at 0.936 ± 0.049. The mean difference is about 0.004, smaller than the variation across fitting seeds. Both give competitive whole-image decisions. The detector also gives defect locations. The classifier gives a whole-image label when boxes are not needed.

Table 6 Visual models on 219 test images

| Model | Accuracy | Balancedaccuracy | Macro F1 | Medianlatency (ms) | 95th-percentilelatency (ms) | Size(MB) |
| --- | --- | --- | --- | --- | --- | --- |
| YOLO11n detector | 0.968± 0.021 | 0.928± 0.056 | 0.940± 0.043 | 12.9± 1.3 | 15.0± 1.9 | 5.5 |
| EfficientNet-B0 | 0.967± 0.025 | 0.918± 0.061 | 0.936± 0.049 | 12.7± 0.1 | 17.6± 1.6 | 16.3 |
| YOLO11n-cls | 0.954± 0.028 | 0.884± 0.077 | 0.908± 0.063 | 9.5± 1.2 | 11.5± 2.4 | 3.2 |
| MobileNetV3-Small | 0.944± 0.014 | 0.853± 0.036 | 0.884± 0.034 | 9.4± 0.2 | 13.8± 0.1 | 6.2 |
| YOLO26n detector | 0.936± 0.024 | 0.856± 0.046 | 0.877± 0.049 | 13.0± 0.1 | 19.3± 4.9 | 5.4 |
| ResNet-18 | 0.935± 0.011 | 0.829± 0.027 | 0.861± 0.029 | 7.2± 0.2 | 8.7± 0.4 | 44.8 |

Values are mean ± sample SD across three fitting seeds. Time includes file reading, input processing, prediction and output processing on the benchmark computer, one image at a time. Size is the mean saved-checkpoint size. Classifiers use 224-pixel inputs; detectors use 640-pixel inputs.

YOLO11n classification has the smallest checkpoint, about 3.2 MB. ResNet-18 has the lowest mean of run median times, about 7.2 ms, even though its checkpoint is 44.8 MB. EfficientNet-B0 and YOLO11n detection have similar times: about 12.7 and 12.9 ms. A smaller checkpoint does not necessarily mean faster inference. Speed also depends on hardware, software, input size and how the model is implemented.

Figure 7(a) ranks image macro F1. Circles, squares and triangles show seeds 17, 29 and 43. The bars start at the displayed 0.50 baseline, not zero. Whiskers show sample standard deviation around the mean. Tied seed points have small vertical offsets so they remain visible. Their horizontal score values do not change. Figure 7(b) compares mean F1 with the mean of each run's median time. Bubble area shows checkpoint size. Model numbers link the panel to the ranking. Figure 7(c) shows AP50 and AP50:95 separately from image classification. Inline legends use shapes, sizes and labels as well as color.

YOLO26n detection reaches mean macro F1 of 0.877 ± 0.049 at about 13.0 ms. With these data and this training budget, the newer detector does not give a better whole-image trade-off. We should test a model update on the actual task. A newer release is not automatically a better model for that task.

Table 7 reports box-location results. YOLO11n reaches overall AP50 of 0.720 ± 0.013 and AP50:95 of 0.421 ± 0.007. AP is higher for underextrusion than for spaghettification. Finding boxes around tangled plastic is harder in this comparison. We would need a separate error review to decide whether labels or defect shape explain the gap. A model can choose the right image class but still miss objects or draw poor boxes. This is why we report both image and box scores.

![Figure 7](../assets/figures/figure_07.png)

Fig. 7 Visual results

Table 7 Defect box results by class

| Detector | Class | Precision | Recall | AP50 | AP50:95 |
| --- | --- | --- | --- | --- | --- |
| YOLO11n | Spaghettification | 0.753± 0.056 | 0.534± 0.015 | 0.588± 0.029 | 0.361± 0.007 |
| YOLO11n | Underextrusion | 0.878± 0.090 | 0.701± 0.028 | 0.852± 0.008 | 0.481± 0.018 |
| YOLO11n | Overall | — | — | 0.720± 0.013 | 0.421± 0.007 |
| YOLO26n | Spaghettification | 0.666± 0.058 | 0.543± 0.033 | 0.566± 0.010 | 0.330± 0.016 |
| YOLO26n | Underextrusion | 0.746± 0.183 | 0.639± 0.082 | 0.785± 0.026 | 0.460± 0.022 |
| YOLO26n | Overall | — | — | 0.675± 0.010 | 0.395± 0.004 |

Values are mean ± sample SD across three fitting seeds. Precision and recall are object-level results, including missed labeled objects in recall. AP50:95 averages IoU thresholds from 0.50 to 0.95. Overall AP averages the two defect classes.

We select YOLO11n detection for the upgraded visual service because the supervisor needs defect locations as well as an image decision. EfficientNet-B0 is the strongest whole-image classifier by mean macro F1, but it has no box output. The small image-score gap does not prove statistical superiority. The main reason for selecting detection is its location output. ResNet-18 and YOLO11n classification offer other speed and storage choices.

### 5.2 Acoustic classification and probability checks

Table 8 compares five acoustic models on the same 168 test rows. The small MLP has the highest mean macro F1, 0.773 ± 0.000, and balanced accuracy, 0.781 ± 0.002. Its sensitivity is 0.833 ± 0.014 and specificity is 0.729 ± 0.010. CatBoost follows with macro F1 of 0.759 ± 0.014. We select the small MLP for the upgraded AE feature service because we use macro F1 as the selection measure. Other models may suit different operating goals.

XGBoost has the highest mean AUROC, about 0.829. Random forest follows at about 0.828. The MLP's higher macro F1 and XGBoost's higher AUROC answer different questions. F1 scores the decisions at a chosen cutoff. AUROC checks score ranking across cutoffs. CatBoost has the lowest mean Brier score. Model selection should consider both class separation and probability quality.

Table 8 Acoustic models on 168 test rows

| Model | Balancedaccuracy | Macro F1 | Sensitivity | Specificity | AUROC |
| --- | --- | --- | --- | --- | --- |
| Small MLP | 0.781± 0.002 | 0.773± 0.000 | 0.833± 0.014 | 0.729± 0.010 | 0.814± 0.004 |
| CatBoost | 0.763± 0.020 | 0.759± 0.014 | 0.773± 0.079 | 0.753± 0.039 | 0.816± 0.002 |
| Logistic regression | 0.762± 0.000 | 0.755± 0.000 | 0.806± 0.000 | 0.719± 0.000 | 0.773± 0.000 |
| Random forest | 0.762± 0.012 | 0.755± 0.011 | 0.806± 0.024 | 0.719± 0.000 | 0.828± 0.002 |
| XGBoost | 0.750± 0.010 | 0.741± 0.016 | 0.806± 0.050 | 0.694± 0.069 | 0.829± 0.003 |
| Model | Brier score | ECE(15 bins) | Latency(ms) | Selectedthreshold |
| Small MLP | 0.177± 0.004 | 0.127± 0.009 | 0.201± 0.005 | 0.247± 0.013 |
| CatBoost | 0.174± 0.002 | 0.129± 0.004 | 0.241± 0.004 | 0.497± 0.119 |
| Logistic regression | 0.189± 0.000 | 0.158± 0.000 | 0.187± 0.002 | 0.325± 0.000 |
| Random forest | 0.175± 0.002 | 0.130± 0.004 | 49.261± 0.526 | 0.475± 0.018 |
| XGBoost | 0.178± 0.001 | 0.126± 0.014 | 0.334± 0.027 | 0.397± 0.136 |

Values are mean ± sample SD across three fitting seeds. ECE uses 15 probability bins. Lower Brier score and ECE are better. Time covers prediction from features already calculated. It excludes sensing and feature calculation.

Figure 8(a) ranks macro F1 in the same 0.50–1.00 window as Figure 7(a). Labels give the mean, sample SD and AUROC. Small vertical offsets keep tied seed symbols visible. Figure 8(b) shows the MLP's ROC curves from saved test probabilities. The curves use matching seed colors and different line patterns. Figure 8(c) gives the seed-17 confusion matrix, a table of correct and wrong decisions. The model correctly labels 69 of 96 normal rows and 61 of 72 defective rows. It has 27 false positives and 11 false negatives. Percentages are calculated within each true class. These counts show the chosen cutoff's trade-off. The pause rule also depends on event order. False-positive rows therefore do not directly tell us how often the printer would be interrupted.

The MLP's mean prediction time is about 0.201 ms for one feature vector already calculated. Logistic regression, XGBoost and CatBoost also have mean times below one millisecond in this test. Random forest takes about 49.261 ms. These results favor small pipelines when features are ready. They do not measure sensing or feature-calculation time, which the complete acoustic service still needs.

![Figure 8](../assets/figures/figure_08.png)

Fig. 8 Acoustic results

## 6 Discussion

### 6.1 Agent tasks and access to records

Manufacturing records are spread across tables, files, live displays and printer services. Operators ask for help in ordinary task language. The supervisor connects these requests to the database and printing agents. A database reply gives records the operator can inspect. A printer reply gives the result of a request or a state query. The workflow keeps these sources clear. A generated answer does not replace a measurement.

Checking the database first is useful when an action depends on saved evidence. The system retrieves the relevant records before proposing a printer action. The operator can inspect the records, time window and job link. Saving the query and result helps the operator check the answer. A lasting interaction log is needed to reconstruct the full request history. The latest displayed result is not enough. Multiple agents also add coordination work. We should compare their value with one agent and a fixed-rule interface on the same tasks.

Agent tests should check routing, SQL execution, record selection, tool inputs, error handling and whether the answer follows the evidence. A valid SQL query can still get the wrong job. A valid API call can still be wrong for the current printer state. Full task success requires all stages to match the operator's request. The workflow figures and tool rules show where to measure these outcomes. They do not give measured success rates.

### 6.2 Data meaning and machine actions

Similar-looking records can mean different things. A visual row is one detection. An AE benchmark row is a set of calculated features. A host insertion timestamp, added when the record is stored, is not a synchronized sensor-event time. A missing row does not prove normal printing. Security checks also differ. Restricting imports, isolating code in a container and allowing a printer command solve different problems. The supervisor must respect these differences when using a tool's reply.

The fusion design links camera and acoustic records by job and time. It gives the supervisor a combined view of a printing interval, including disagreement or missing data. A visible defect may remain after a short acoustic event ends. Acoustic activity may also happen before a visible defect. We therefore keep individual observations instead of immediately combining them into one confidence score.

Real fusion testing needs both sensors to record the same jobs with shared event labels. The test should compare each sensor alone, agreement-based rules and any learned combination under matching operating limits. It should also check delayed, missing and old records. This paper defines the links and supervisory interfaces. It does not report measured fused-sensing performance.

Actions that affect the printer need checks outside the LLM. A validator should check allowed operations, input limits, current-job identity and record age. Approval rules should match the action's effect. The logs should separately record the requested action, its acceptance and the physical outcome. This is important for temperature changes, movement and pauses. A plausible reply does not prove correct execution.

### 6.3 What the recognition tests support

The recognition tests help choose the services that give evidence to the supervisor. YOLO11n detection and EfficientNet-B0 have similar image scores. The detector also returns regions. An operator asking where a defect is needs location information. A simple image alarm may only need classification. File size and processing time describe different limits. Both should be checked on the intended deployment hardware.

For AE, the small MLP gives competitive results on six features with low prediction time. The tree models' stronger AUROC may matter when the goal is good ranking or high sensitivity at a chosen false-alarm rate. A deployed model must keep its fitted estimator, input processing, feature order, class map and cutoff together. Changing just one of these can change the meaning of the result shown to the agent or operator.

Recognition accuracy does not directly show a benefit to manufacturing. Consecutive-observation rules, sampling intervals, communication and printer behavior decide how a detection becomes an action. Useful measures include false pauses per printing hour, detection delay, accepted parts and material use. We should log acquisition time, host receipt, model completion, policy checks, command acknowledgment and observed printer state. These records would connect model scores to actual printing outcomes.

### 6.4 Limits and the next tests

The tests use fixed image and feature-row splits. The visual data have three sequences, and classes are linked to sequences. The acoustic data have no job grouping or complete acquisition history. Three fitting seeds show variation in training. They do not create independent machines, materials or print jobs. Testing transfer to new shapes, lighting, materials and sensor mounts requires separate data grouped by job or recording condition.

We have not measured agent-task accuracy, coordination cost, independently enforced command safety or physical fusion. Hardware ratings describe device capacity. The feature archive does not give complete gain, filter or sensor-mount temperature settings. The reported computer timings do not establish Raspberry Pi inference speed or printer stopping time. The implemented interfaces and recognition results therefore remain separate from the full operating outcomes still to be tested.

The next tests should first hold the agent settings, tool rules, recognition models and decision rules constant. We should then run the tasks in Section 4.6. Record retrieval and tool-call success should be measured separately before full task success. Printer traces should show the requested command, its acknowledgment and the state observed independently afterward. Separate print jobs with synchronized sensors can then support event-level fusion tests and false-pause counts per printing hour. Sensor settings, command outcomes and part-quality checks would show how agent correctness affects manufacturing.

## 7 Conclusion

We presented a local AI agent system that connects FFF operator requests to process records and printer services. A supervisor coordinates database and printing agents through defined tools. Returned observations, accepted commands and physically confirmed outcomes stay separate. Sensor services can request a pause directly. The data design links image detections, acoustic events and printer readings by job and time. These links help the operator inspect a printing interval.

Supporting tests compare six visual models and five acoustic models on fixed data splits. YOLO11n detection reaches image macro F1 of 0.940 ± 0.043. EfficientNet-B0 reaches 0.936 ± 0.049. YOLO11n also returns boxes with AP50 of 0.720 ± 0.013. The small acoustic MLP reaches macro F1 of 0.773 ± 0.000. We choose detection for its location output and the MLP for its feature-level decisions. The AMD computer is assigned the local language-model deployment role. NVIDIA hardware supports visual training. This paper defines the agent workflow and supports sensing-model selection. Full system testing still needs agent-task measurements, enforced code isolation and synchronized job-level recordings.

## Statements and declarations

### Data and code availability

The benchmark code, archived AE feature data, frozen split lists, model settings, saved predictions, training logs, figures and reproduction instructions are available in the consolidated evidence repository: [Consolidated evidence repository](https://github.com/bensidhom/Agentic-Multimodal-3D-Printing). Raw external images are obtained through the named dataset provider under its access and reuse conditions; large visual checkpoints are not bundled. The repository records software versions, provenance and reuse conditions. The original system and acquisition implementations remain available in [MultiAgent SCADA](https://github.com/bensidhom/MultiAgent_SCADA), [AI-driven analytics and digital twins](https://github.com/bensidhom/Novel-Cyber-Infrastructure-Enabling-AI-Driven-Analytics-and-Digital-Twins-for-Real-Time-3D-Printing) and [Acoustic emission monitoring at the edge](https://github.com/bensidhom/Acoustic-Emission-Monitoring-at-the-Edge-for-Real-Time-Artificial-Intelligence-Applications).

### Research ethics

The study uses manufacturing records and software outputs. It does not collect measurements from human participants.

## References

AMD. (n.d.). AMD Ryzen AI Max+ 395 product specifications. *Retrieved October 2*, 2026, from [https://www.amd.com/en/products/processors/laptop/ryzen/ai-300-series/amd-ryzen-ai-max-plus-395.html](https://www.amd.com/en/products/processors/laptop/ryzen/ai-300-series/amd-ryzen-ai-max-plus-395.html)

Banadaki, Y., Razaviarab, N., Fekrmandi, H., Li, G., Mensah, P., Bai, S., & Sharifi, S. (2022). Automated quality and process control for additive manufacturing using deep convolutional neural networks. *Recent Progress in Materials*, 04, 1–19. [https://doi.org/10.21926/rpm.2201005](https://doi.org/10.21926/rpm.2201005)

Ben Sidhom, A. (n.d.-a). *Acoustic Emission Monitoring at the Edge for Real-Time Artificial Intelligence Applications [Software and acquisition repository]*. GitHub. [https://github.com/bensidhom/Acoustic-Emission-Monitoring-at-the-Edge-for-Real-Time-Artificial-Intelligence-Applications](https://github.com/bensidhom/Acoustic-Emission-Monitoring-at-the-Edge-for-Real-Time-Artificial-Intelligence-Applications)

Ben Sidhom, A. (n.d.-b). *MultiAgent_SCADA [Software and data repository]*. GitHub. [https://github.com/bensidhom/MultiAgent_SCADA](https://github.com/bensidhom/MultiAgent_SCADA)

Ben Sidhom, A. (n.d.-c). *Novel Cyber Infrastructure Enabling AI-Driven Analytics and Digital Twins for Real-Time 3D Printing [Software and data repository]*. GitHub. [https://github.com/bensidhom/Novel-Cyber-Infrastructure-Enabling-AI-Driven-Analytics-and-Digital-Twins-for-Real-Time-3D-Printing](https://github.com/bensidhom/Novel-Cyber-Infrastructure-Enabling-AI-Driven-Analytics-and-Digital-Twins-for-Real-Time-3D-Printing)

Brion, D. A. J., & Pattinson, S. W. (2022). Generalisable 3D printing error detection and correction via multi-head neural networks. *Nature Communications*, 13, 4654. [https://doi.org/10.1038/s41467-022-31985-y](https://doi.org/10.1038/s41467-022-31985-y)

Charalampous, P., Kostavelis, I., Kopsacheilis, C., & Tzovaras, D. (2021). Vision-based real-time monitoring of extrusion additive manufacturing processes for automatic manufacturing error detection. *The International Journal of Advanced Manufacturing Technology*, 115, 3859–3872. [https://doi.org/10.1007/s00170-021-07419-2](https://doi.org/10.1007/s00170-021-07419-2)

Chen, L., Yao, X., Liu, K., Tan, C., & Moon, S. K. (2023). Multisensor fusion-based digital twin in additive manufacturing for in-situ quality monitoring and defect correction. *Proceedings of the Design Society*, 3, 2755–2764. [https://doi.org/10.1017/pds.2023.276](https://doi.org/10.1017/pds.2023.276)

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794. [https://doi.org/10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785)

Elbasheer, M., Laili, Y., Longo, F., Solina, V., Tao, Y., Veltri, P., Zhang, Y., & Zhang, L. (2026). Natural language-driven production planning: integrating large language models with automatic simulation model generation in manufacturing systems. *Journal of Intelligent Manufacturing*, 37, 4431–4458. [https://doi.org/10.1007/s10845-025-02732-z](https://doi.org/10.1007/s10845-025-02732-z)

Forster, R., Soulioti, D., Feteira, A., Grammatikos, S., & Kordatos, E. (2026). In-line monitoring of the material extrusion additive manufacturing process by non-destructive methodologies. *The International Journal of Advanced Manufacturing Technology*. [https://doi.org/10.1007/s00170-026-19170-7](https://doi.org/10.1007/s00170-026-19170-7)

Fu, Y., Downey, A. R. J., Yuan, L., & Huang, H. T. (2023). Real-time structural validation for material extrusion additive manufacturing. *Additive Manufacturing*, 65, 103409. [https://doi.org/10.1016/j.addma.2023.103409](https://doi.org/10.1016/j.addma.2023.103409)

Fu, Y., Downey, A., Yuan, L., Pratt, A., & Balogun, Y. (2021). In situ monitoring for fused filament fabrication process: A review. *Additive Manufacturing*, 38, 101749. [https://doi.org/10.1016/j.addma.2020.101749](https://doi.org/10.1016/j.addma.2020.101749)

Großeheide, J., Tao, Z., Cetin, B., Wolfschläger, D., Dewulf, W., & Schmitt, R. H. (2025). Enabling multimodal sensor fusion in additive manufacturing using a voxel-based data model. *Production Engineering*, 19, 1195–1216. [https://doi.org/10.1007/s11740-025-01364-9](https://doi.org/10.1007/s11740-025-01364-9)

Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. *Proceedings of Machine Learning Research*, 70, 1321–1330. [https://proceedings.mlr.press/v70/guo17a.html](https://proceedings.mlr.press/v70/guo17a.html)

He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 770–778. [https://doi.org/10.1109/CVPR.2016.90](https://doi.org/10.1109/CVPR.2016.90)

Holland, M., & Chaudhari, K. (2024). Large language model based agent for process planning of fiber composite structures. *Manufacturing Letters*, 40, 100–103. [https://doi.org/10.1016/j.mfglet.2024.03.010](https://doi.org/10.1016/j.mfglet.2024.03.010)

Holzmond, O., & Li, X. (2017). In situ real time defect detection of 3D printed parts. *Additive Manufacturing*, 17, 135–142. [https://doi.org/10.1016/j.addma.2017.08.003](https://doi.org/10.1016/j.addma.2017.08.003)

Howard, A., Sandler, M., Chu, G., Chen, L.-C., Chen, B., Tan, M., Wang, W., Zhu, Y., Pang, R., Vasudevan, V., Le, Q. V., & Adam, H. (2019). Searching for MobileNetV3. *Proceedings of the IEEE/CVF International Conference on Computer Vision*, 1314–1324. [https://doi.org/10.1109/ICCV.2019.00140](https://doi.org/10.1109/ICCV.2019.00140)

Hugging Face. (n.d.). Secure code execution. *smolagents documentation. Retrieved October 1*, 2026, from [https://huggingface.co/docs/smolagents/v1.22.0/en/tutorials/secure_code_execution](https://huggingface.co/docs/smolagents/v1.22.0/en/tutorials/secure_code_execution)

Hugging Face. (2025). *smolagents version 1.22.0: Agent implementation*. [https://github.com/huggingface/smolagents/blob/v1.22.0/src/smolagents/agents.py](https://github.com/huggingface/smolagents/blob/v1.22.0/src/smolagents/agents.py)

Jadhav, Y., Pak, P., & Barati Farimani, A. (2025). LLM-3D print: Large Language Models to monitor and control 3D printing. *Additive Manufacturing*, 114, 105027. [https://doi.org/10.1016/j.addma.2025.105027](https://doi.org/10.1016/j.addma.2025.105027)

Jones, D., Snider, C., Nassehi, A., Yon, J., & Hicks, B. (2020). Characterising the Digital Twin: A systematic literature review. *CIRP Journal of Manufacturing Science and Technology*, 29, 36–52. [https://doi.org/10.1016/j.cirpj.2020.02.002](https://doi.org/10.1016/j.cirpj.2020.02.002)

Kapoor, S., & Narayanan, A. (2023). Leakage and the reproducibility crisis in machine-learning-based science. *Patterns*, 4, 100804. [https://doi.org/10.1016/j.patter.2023.100804](https://doi.org/10.1016/j.patter.2023.100804)

Kritzinger, W., Karner, M., Traar, G., Henjes, J., & Sihn, W. (2018). Digital Twin in manufacturing: A categorical literature review and classification. *IFAC-PapersOnLine*, 51, 1016–1022. [https://doi.org/10.1016/j.ifacol.2018.08.474](https://doi.org/10.1016/j.ifacol.2018.08.474)

Kumar, S., Gopi, T., Harikeerthana, N., Gupta, M. K., Gaur, V., Krolczyk, G. M., & Wu, C. (2023). Machine learning techniques in additive manufacturing: a state of the art review on design, processes and production control. *Journal of Intelligent Manufacturing*, 34, 21–55. [https://doi.org/10.1007/s10845-022-02029-5](https://doi.org/10.1007/s10845-022-02029-5)

Li, H., Yu, Z., Li, F., Kong, Q., & Tang, J. (2022). Real-time polymer flow state monitoring during fused filament fabrication based on acoustic emission. *Journal of Manufacturing Systems*, 62, 628–635. [https://doi.org/10.1016/j.jmsy.2022.01.007](https://doi.org/10.1016/j.jmsy.2022.01.007)

Lim, J., Vogel-Heuser, B., & Kovalenko, I. (2024). Large Language Model-Enabled Multi-Agent Manufacturing Systems. *2024 IEEE 20th International Conference on Automation Science and Engineering (CASE)*, 3940–3946. [https://doi.org/10.1109/case59546.2024.10711432](https://doi.org/10.1109/case59546.2024.10711432)

Lin, W., Shen, H., Fu, J., & Wu, S. (2019). Online quality monitoring in material extrusion additive manufacturing processes based on laser scanning technology. *Precision Engineering*, 60, 76–84. [https://doi.org/10.1016/j.precisioneng.2019.06.004](https://doi.org/10.1016/j.precisioneng.2019.06.004)

Liu, J., Hu, Y., Wu, B., & Wang, Y. (2018). An improved fault diagnosis approach for FDM process with acoustic emission. *Journal of Manufacturing Processes*, 35, 570–579. [https://doi.org/10.1016/j.jmapro.2018.08.038](https://doi.org/10.1016/j.jmapro.2018.08.038)

Logitech. (n.d.). C270 HD webcam specifications. *Retrieved October 2*, 2026, from [https://www.logitech.com/en-us/shop/p/c270-hd-webcam.960-000694](https://www.logitech.com/en-us/shop/p/c270-hd-webcam.960-000694)

Lu, Q. Y., & Wong, C. H. (2018). Additive manufacturing process monitoring and control by non-destructive testing techniques: challenges and in-process monitoring. *Virtual and Physical Prototyping*, 13, 39–48. [https://doi.org/10.1080/17452759.2017.1351201](https://doi.org/10.1080/17452759.2017.1351201)

Mermaid. (n.d.). Entity relationship diagrams. *Retrieved October 4*, 2026, from [https://mermaid.js.org/syntax/entityRelationshipDiagram.html](https://mermaid.js.org/syntax/entityRelationshipDiagram.html)

OctoPrint Project. (n.d.). Job operations. *OctoPrint REST API documentation. Retrieved October 1*, 2026, from [https://docs.octoprint.org/en/main/api/job.html](https://docs.octoprint.org/en/main/api/job.html)

Oleff, A., Küster, B., Stonis, M., & Overmeyer, L. (2021). Process monitoring for material extrusion additive manufacturing: a state-of-the-art review. *Progress in Additive Manufacturing*, 6, 705–730. [https://doi.org/10.1007/s40964-021-00192-4](https://doi.org/10.1007/s40964-021-00192-4)

Ollama. (n.d.-a). Hardware support. *Retrieved October 2*, 2026, from [https://docs.ollama.com/gpu](https://docs.ollama.com/gpu)

Ollama. (n.d.-b). Qwen3.6 35B model library entry. *Retrieved October 2*, 2026, from [https://ollama.com/library/qwen3.6:35b](https://ollama.com/library/qwen3.6:35b)

Ollama. (n.d.-c). Qwen3-Coder 30B model library entry. *Retrieved October 2*, 2026, from [https://ollama.com/library/qwen3-coder:30b](https://ollama.com/library/qwen3-coder:30b)

Petsiuk, A. L., & Pearce, J. M. (2020). Open source computer vision-based layer-wise 3D printing analysis. *Additive Manufacturing*, 36, 101473. [https://doi.org/10.1016/j.addma.2020.101473](https://doi.org/10.1016/j.addma.2020.101473)

Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., & Gulin, A. (2018). CatBoost: Unbiased boosting with categorical features. *Advances in Neural Information Processing Systems*, 31. [https://papers.nips.cc/paper_files/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html](https://papers.nips.cc/paper_files/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html)

Qin, J., Hu, F., Liu, Y., Witherell, P., Wang, C. C. L., Rosen, D. W., Simpson, T. W., Lu, Y., & Tang, Q. (2022). Research and application of machine learning for additive manufacturing. *Additive Manufacturing*, 52, 102691. [https://doi.org/10.1016/j.addma.2022.102691](https://doi.org/10.1016/j.addma.2022.102691)

Qwen Team. (2026). *Qwen3.6-35B-A3B model card*. [https://huggingface.co/Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)

Rachmawati, S. M., Putra, M. A. P., Lee, J. M., & Kim, D. S. (2023). Digital twin-enabled 3D printer fault detection for smart additive manufacturing. *Engineering Applications of Artificial Intelligence*, 124, 106430. [https://doi.org/10.1016/j.engappai.2023.106430](https://doi.org/10.1016/j.engappai.2023.106430)

Raspberry Pi Ltd. (n.d.). Raspberry Pi 4 Model B specifications. *Retrieved October 2*, 2026, from [https://www.raspberrypi.com/products/raspberry-pi-4-model-b/specifications/](https://www.raspberrypi.com/products/raspberry-pi-4-model-b/specifications/)

Roboflow. (n.d.). Image annotation and dataset management. *Retrieved October 2*, 2026, from [https://roboflow.com/annotate](https://roboflow.com/annotate)

Roeder, L. (n.d.). Netron [Model visualization software]. *Retrieved October 4*, 2026, from [https://github.com/lutzroeder/netron](https://github.com/lutzroeder/netron)

Rooney, S., Pitz, E., & Pochiraju, K. (2025). AutoML-driven diagnostics of the feeder motor in fused filament fabrication machines from direct current signals. *Journal of Intelligent Manufacturing*, 36, 1999–2016. [https://doi.org/10.1007/s10845-024-02332-3](https://doi.org/10.1007/s10845-024-02332-3)

Sadaf, A. I., & Ahmed, H. (2026). Spatiotemporal deep learning using infrared thermography for real-time monitoring and process parameter adjustment in material extrusion. *Journal of Intelligent Manufacturing*, 37, 3713–3732. [https://doi.org/10.1007/s10845-025-02702-5](https://doi.org/10.1007/s10845-025-02702-5)

Schall, D. (2026). Safe integration of Large Language Models into industrial process control: a multi-agent architecture with P&ID-grounded validation. *Autonomous Intelligent Systems*, 6, 14. [https://doi.org/10.1007/s43684-026-00136-1](https://doi.org/10.1007/s43684-026-00136-1)

Straub, J. (2015). Initial work on the characterization of additive manufacturing (3D printing) using software image analysis. *Machines*, 3, 55–71. [https://doi.org/10.3390/machines3020055](https://doi.org/10.3390/machines3020055)

Tan, M., & Le, Q. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. *Proceedings of Machine Learning Research*, 97, 6105–6114. [https://proceedings.mlr.press/v97/tan19a.html](https://proceedings.mlr.press/v97/tan19a.html)

Tao, F., Zhang, H., Liu, A., & Nee, A. Y. C. (2019). Digital twin in industry: State-of-the-art. *IEEE Transactions on Industrial Informatics*, 15, 2405–2415. [https://doi.org/10.1109/tii.2018.2873186](https://doi.org/10.1109/tii.2018.2873186)

Ultralytics. (n.d.-a). Object detection with Ultralytics YOLO. *Retrieved October 2*, 2026, from [https://docs.ultralytics.com/tasks/detect/](https://docs.ultralytics.com/tasks/detect/)

Ultralytics. (n.d.-b). Ultralytics YOLO26. *Documentation. Retrieved October 1*, 2026, from [https://docs.ultralytics.com/models/yolo26/](https://docs.ultralytics.com/models/yolo26/)

Ultralytics. (n.d.-c). YOLO11 model configuration. *Retrieved October 4*, 2026, from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/11/yolo11.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/11/yolo11.yaml)

Vallen Systeme. (n.d.-a). *VS600-Z1 piezoelectric AE sensor: Technical data*. [https://www.vallen.de/wp-content/uploads/2019/03/VS600-Z1_1512.pdf](https://www.vallen.de/wp-content/uploads/2019/03/VS600-Z1_1512.pdf)

Vallen Systeme. (n.d.-b). WaveLine products: spotWave 201. *Retrieved October 1*, 2026, from [https://www.vallen.de/products/data-acquisiton-units/waveline-products/](https://www.vallen.de/products/data-acquisiton-units/waveline-products/)

Vallen Systeme. (n.d.-c). *WaveLine spotWave acquisition implementation [Software]*. [https://github.com/vallen-systems/waveline/blob/70b56d4387e7e58f45ffe4662ed51da83ef09709/src/waveline/spotwave.py](https://github.com/vallen-systems/waveline/blob/70b56d4387e7e58f45ffe4662ed51da83ef09709/src/waveline/spotwave.py)

Vyas, J., & Mercangöz, M. (2025). Autonomous Industrial Control using an Agentic Framework with Large Language Models. *IFAC-PapersOnLine*, 59, 349–354. [https://doi.org/10.1016/j.ifacol.2025.07.170](https://doi.org/10.1016/j.ifacol.2025.07.170)

Waheed, M. F., Bernadin, S., Hassan, A., & Francois, J. (2025). Real Time Fault Detection in 3D Printers Using Convolutional Neural Networks and Acoustic Signals. *SoutheastCon 2025*, 1294–1299. [https://doi.org/10.1109/southeastcon56624.2025.10971647](https://doi.org/10.1109/southeastcon56624.2025.10971647)

Xu, K., Lyu, J., & Manoochehri, S. (2022). In situ process monitoring using acoustic emission and laser scanning techniques based on machine learning models. *Journal of Manufacturing Processes*, 84, 357–374. [https://doi.org/10.1016/j.jmapro.2022.10.002](https://doi.org/10.1016/j.jmapro.2022.10.002)

Yang, Z., Jin, L., Yan, Y., & Mei, Y. (2018). Filament breakage monitoring in fused deposition modeling using acoustic emission technique. *Sensors*, 18, 749. [https://doi.org/10.3390/s18030749](https://doi.org/10.3390/s18030749)

Yu, B., Zhang, H., Tang, J., Zhao, Z., Zhng, X., He, W., & Chen, X. (2025). Multi-view deep information fusion framework for online quality monitoring and autonomous correction in material extrusion additive manufacturing. *Virtual and Physical Prototyping*, 20, e2500672. [https://doi.org/10.1080/17452759.2025.2500672](https://doi.org/10.1080/17452759.2025.2500672)

Zhao, Z., Tang, D., Liu, C., Wang, L., Zhang, Z., Zhu, H., Chen, K., Nie, Q., & Ji, Y. (2026). A Large language model-based multi-agent manufacturing system for intelligent shopfloors. *Advanced Engineering Informatics*, 69, 103888. [https://doi.org/10.1016/j.aei.2025.103888](https://doi.org/10.1016/j.aei.2025.103888)
