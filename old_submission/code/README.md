## Tasks Accomplished

- [x] **Task 1:** Surveillence of current traffic control systems deployed in India
- [x] **Task 2:** Mapping roads lanes and intersections near Connaught Place, Delhi
- [x] **Task 3:** Optimizing Traffic control systems on the mapped intersections

## Technology Stack

This project leverages the following technologies:

- **[Eclipse SUMO](https://eclipse.dev/sumo/):** Eclipse SUMO (Simulation of Urban MObility) is an open source, highly portable, microscopic and continuous multi-modal traffic simulation package designed to handle large networks.
- **[Open CV](https://opencv.org/):** Used to track and map day to day vehicular flow near Connaught place for training our QMIX model.
- **[Flask](https://flask.palletsprojects.com/en/stable/):** Used for implementing backend logic for the simulation. 

## Key Features

- **Feature 1:** Increases the efficiency of the currently deployed traffic light control systems in India.
- **Feature 2:** Provides a dashboard for traffic authorities to monitor and control traffic intersections remotely.
- **Feature 3:** Uses Open CV for real time traffic load management.

## Project Structure
```
code
├── main.py
├── SUMO
├── maps
│   ├── connaught_place.net.xml
│   ├── connaught_place.rou.xml
│   ├── connaught_place.sumocfg
│   ├── gui_settings.xml
│   ├── README.md
│   └── routes.rou.xml
├── models
│   ├── qmix_agent.pth
│   └── qmix_mixing.pth
├── README.md
├── requirements.txt
├── results.txt
├── run_sumo.bat
├── src
│   ├── algorithms
│   │   ├── __pycache__
│   │   │   ├── qmix_net.cpython-312.pyc
│   │   │   └── qmix_trainer.cpython-312.pyc
│   │   ├── qmix_net.py
│   │   └── qmix_trainer.py
│   ├── asd.txt
│   ├── config
│   │   ├── __init__.py
│   │   └── sumo_qmix.yaml
│   ├── dual_simulation_runner.py
│   ├── envs
│   │   ├── __init__.py
│   │   ├── multiagentenv.py
│   │   ├── __pycache__
│   │   │   ├── __init__.cpython-312.pyc
│   │   │   ├── __init__.cpython-313.pyc
│   │   │   ├── multiagentenv.cpython-312.pyc
│   │   │   ├── multiagentenv.cpython-313.pyc
│   │   │   ├── sumo_env.cpython-312.pyc
│   │   │   ├── SUMOEnv.cpython-312.pyc
│   │   │   └── sumo_env.cpython-313.pyc
│   │   └── sumo_env.py
│   ├── evaluate_policy.py
│   ├── main.py
│   ├── profile_time.py
│   ├── __pycache__
│   │   ├── dual_simulation_runner.cpython-313.pyc
│   │   ├── main.cpython-312.pyc
│   │   ├── qmix_models.cpython-312.pyc
│   │   ├── qmix_models.cpython-313.pyc
│   │   └── replay_buffer.cpython-312.pyc
│   ├── qmix_models.py
│   ├── replay_buffer.py
│   ├── run.py
│   ├── run_sumo_gui.py
│   └── train_qmix.py
└── templates
    ├── about.html
    ├── contact.html
    ├── dashboard.html
    ├── features.html
    ├── index.html
    └── simulation.html

10 directories, 50 files
```

## Local Setup Instructions (Write for both windows and macos)

Follow these steps to run the project locally

1. **Clone the Repository**
   ```bash
   git clone https://github.com/TeamDUI/Traffic-Management-System.git
   cd Traffic-Management-System
   cd code
   ```

2. **Installation**  
    To set up the project, clone the repository and install the required dependencies listed in `requirements.txt`.

    ```bash
    pip install -r requirements.txt
    ```   

3. **Usage**
    1. Use flask to deploy the website to the localhost
    2. To run the simulation open the link http://127.0.0.1:5000/, navigate to the simulation tab and click the 'run simulation' button

    ```bash
    python main.py
    ```

    Note: This simulation only runs on Windows as SUMO is not supported on any other operating system.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
