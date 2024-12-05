<h2 align="center" style="margin-top: -1.5cm;">
  <strong>Models for scalability and precision evaluation of GPU-based N-Body Approximations</strong>
</h2>

This work investigates the behavior of *N-Body* simulations using the *Barnes-Hut* algorithm on GPUs, aiming to establish predictive models to estimate execution times and simulation errors based on various hardware configurations and execution parameters, leveraging machine learning techniques. An exhaustive analysis and profiling of executions are conducted across multiple systems, collecting hardware specifications, profiling metrics, and simulation parameters, which are filtered and categorized through *Feature Extraction*. 

Subsequently, five types of models —*Linear Regression*, *Polynomial Regression*, *Support Vector Regression*, *Decision Tree Regression*, and *Extremely Randomized Tree Regression*— are trained and evaluated using a preprocessed and transformed subset of data, applying different *Cross-Validation* techniques to ensure model robustness. 

Finally, a software tool is developed to input hardware characteristics and execution configurations, generating graphical projections of the expected performance. This tool is used both to validate the models and to offer a public resource for evaluating the behavior of the trained *Barnes-Hut* implementation. 

This research aims to contribute to the development of more efficient scientific simulations on GPUs, addressing the lack of studies on execution analysis and its variation based on available computational resources. All generated materials are intended to be openly published, promoting continuous iteration and improvement based on the research results.

---

**Keywords:**  
N-Body; Barnes-Hut; CUDA; Machine Learning;
<p align="center">
  <a href="https://github.com/Mauricio-Bernuy/Bonsai-estimate/blob/master/doc/Mauricio%20Bernuy%20-%20Modelos%20para%20evaluaci%C3%B3n%20de%20escalabilidad%20y%20precisi%C3%B3n%20de%20Aproximaciones%20N-Body%20en%20GPU.pdf">
    <img width="50%" src="https://raw.githubusercontent.com/Mauricio-Bernuy/Bonsai-estimate/refs/heads/master/doc/title.png" alt="Title Image">
  </a>
</p>

## Run visualizer
```
streamlit run "9 Model Visualizer.py"
```
