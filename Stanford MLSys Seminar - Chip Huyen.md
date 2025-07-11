# Stanford MLSys Seminar: Principles of Good Machine Learning Systems Design - Chip Huyen
Just watched an awesome talk by Chip Huyen on Principles of Good Machine Learning Systems Design from the Stanford MLSys Seminar series, and let me tell you, it's a goldmine for anyone looking to bridge the gap between ML research and production

It was a truly insightful talk, packed with practical advice and a clear roadmap for navigating the complexities of ML in the real world. Highly recommend checking it out: [Stanford MLSys Seminar Episode 5](https://www.youtube.com/watch?v=c_AUuTuPA5k&t=1581s&ab_channel=StanfordMLSysSeminars)

Let's dive into some of the key takeaways

### Table of Contents
0. [Abstract](#abstract)
1. [ML in Research vs. Production: A Tale of Two Worlds](#part-1)
2. [ML Production Myths Debunked](#part-2)
3. [The Iterative ML Process](#part-3)

    3.1. [A Blueprint for ML](#part-3.1)

    3.2. [Understanding the Fluid Roles: Data Science vs. ML Engineering](#part-3.2)

    3.3. [The Iterative Process Stages: A Deep Dive](#part-3.3)

    3.4. [Key Insights on ML Engineering](#part-3.4)

4. [Phases of Adoption](#part-4)
5. [Q&A Highlights: Deeper Dives](#part-5)

## Abstract <a name="abstract"></a>
This talk covers what it means to operationalize ML models. It starts by analyzing the difference between ML in research vs. in production, ML systems vs. traditional software, as well as myths about ML production.

It then goes over the principles of good ML systems design and introduces an iterative framework for ML systems design, from scoping the project, data management, model development, deployment, maintenance, to business analysis. It covers the differences between DataOps, ML Engineering, MLOps, and data science, and where each fits into the framework. It also discusses the main skills each stage requires, which can help companies in structuring their teams.

The talk ends with a survey of the ML production ecosystem, the economics of open source, and open-core businesses.

## 1. ML in Research vs. Production: A Tale of Two Worlds <a name="part-1"></a>

Chip kicked things off by highlighting the fundamental differences between ML in a research setting and in production. It's not just about bigger models.

*<p style="text-align:center;">ML in Research vs. Production</p>*

| |Research |Production |
|-|---------|-----------|
|Objectives|Model performance |Different stakeholders have different objectives |
|Computational priority|Fast training, high throughput |Fast inference, low latency |
|Data |Static |Constantly Shifting |
|Fairness |Good to have (sadly) |Important |
|Interpretability |Good to have |Important |

*   **Objectives:** In research, it's usually about **clear, singular objectives**, often focused on achieving the highest model performance on a specific dataset or leaderboard. Think pushing the accuracy needle! But in production? It's a complex balancing act with **multiple stakeholders**. The ML team might want accuracy, the product team demands fast inference (because **"latency costs a lot of money!"**), sales wants to boost ads, and managers are all about maximizing profit. You've got to build systems that satisfy *all* these diverse goals.
*   **Computational Priority:** In research, you're often training models many, many times, so the focus is on **high throughput** to make the most of your hardware during training. In contrast, in production, you might train a model once, but you'll serve it *countless* times. This means the priority shifts dramatically to **fast inference (low latency)**. Chip used a super neat analogy of an ant carrying leaves: 
    + latency is how long it takes to carry one leaf
    + throughput is how many leaves it carries over time
    
    It gets complex when you're serving multiple customers, as higher latency can surprisingly lead to higher throughput if you're batching requests – but in production, **real-time inference** is often preferred because even a small increase in latency can significantly reduce conversion rates and revenue.

*   **Data & Other Factors:** Research often uses **static, standardized datasets**, while production data is **dynamic and can change rapidly** due to things like new marketing campaigns or shifting user demographics. Also, while **fairness and interpretability** are increasingly pursued in research, they're often still niche. In production, however, especially in critical applications like medicine, these become **extremely important characteristics** of the model itself, not just an afterthought.

## 2. ML Production Myths Debunked <a name="part-2"></a>

Chip also tackled some common misconceptions about ML in production:

*   **Myth 1: Deploying is Hard.** Chip says **deploying a model is actually pretty easy** with modern frameworks – you can create an endpoint and build a simple app in hours. The *real* challenge is **deploying it reliably**: keeping it up, minimizing latency, getting alerted to issues, debugging, and rolling back updates. It's about building the entire infrastructure around it for monitoring and maintenance.
*   **Myth 2: You Only Deploy One or Two ML Models.** Nope! Services and apps often require **many different ML models**. Think about Uber serving in 100 countries, potentially needing different models for different demographics, leading to hundreds or even thousands of models in production. Netflix, Booking.com (150+ models!), and Uber (thousands!) are prime examples.
*<p style="text-align:center;">ML models deployed in Netflix</p>*
![NetFlix ML models deployed](images/netflix-ml-models.png)

*   **Myth 3: If We Don't Do Anything, Model Performance Remains the Same**. Software generally suffers from "bit rot" – it decays over time. ML models are even more susceptible due to **concept drift**, where the underlying data distribution or task requirements change. A sentiment analysis model might need to adapt if the company suddenly wants to classify anger instead of just positive/negative. Chip recommends regularly evaluating model performance on current data compared to data from months ago to understand this degradation.
*   **Myth 4: You Don't Need to Update Models Much.** This ties into concept drift. Project managers often budget for initial development but overlook continuous updates. Chip suggests a **20/80 rule**: 20% investment in initial development, 80% in **iterative development and improvement** afterward. The key question isn't *how often you should* update, but **how often you *can*** update. Companies like Netflix deploy thousands of times a day, and AWS every 11 seconds! Chip believes that if ML engineers understood Devops better, the world (or at least ML systems) would be a much better place!
*   **Myth 5: Scale Isn't an Issue for Most Companies.** While many companies start small, over 50% of engineers work for companies with 100+ employees, meaning most engineers *will* have to think about scale. So, it's wise to consider it!
*<p style="text-align:center;">Distribution of Company Size</p>*
![Company Size](images/company-size.png)
*   **Myth 6: ML Can Magically Transform Business Overnight.** AI isn't a snake oil! It takes time and maturity. Only a small percentage of companies investing in ML see immediate returns, often because they're in early stages. Companies like Google, who have seen incredible transformations (like 100% of English queries processed by BERT in 2020), have been investing in AI for many years. The more mature a company's ML pipeline, the faster it can bring models into production, reducing engineering costs and improving support. Managing expectations is crucial!
*<p style="text-align:center;">Efficiency improves with maturity</p>*
![Model deployment timeline and ML maturity](images/deployment-maturity.png)

## 3. The Iterative ML Process <a name="part-3"></a>

Chip introduced an **iterative process for ML production**, covering stages from project scoping to business analysis. She also touched on the often-confusing roles of data science and ML engineering, noting that roles can be very fluid across companies (e.g., Uber's ML team focusing on infrastructure, data science on business insights).
*<p style="text-align:center;">The Iterative ML Process</p>*
![Iterative ML process](images/iterative-ml-process.png)

*This diagram breaks down the different stages of the ML production pipeline, moving beyond just the model itself to encompass the entire lifecycle.*

### 3.1. A Blueprint for ML Production <a name="part-3.1"></a>

Chip's diagram is structured in layers to illustrate the complexity:

*   **Inner Layer:** Represents the **different stages** in the pipeline.
*   **Middle Layer:** Focuses on the **confusing, yet common, terminology** like DataOps, MLOps, ML Engineering, and Data Science, highlighting where each broadly fits in.
*   **Outermost Layer:** Indicates the **main skills required** at each stage.

While the diagram shows a single arrow between stages, Chip emphasizes that in reality, it's "a lot more complicated than that" with "a lot more arrows" and interconnectedness.

### 3.2. Understanding the Fluid Roles: Data Science vs. ML Engineering <a name="part-3.2"></a>

One key area of confusion Chip addresses is the distinction between Data Science and Machine Learning Engineering, noting that **roles can be very fluid across companies**.

*   **Data Science:** Often focuses on using statistics to **generate business insights** from data. Chip hypothesizes that many data scientists transitioned into modeling roles due to existing in-house teams and company requirements before dedicated ML teams were common.
*   **Machine Learning Engineering:** Primarily aims to **build machine learning products**.

Chip highlights Uber as an example where data scientists generate business insights, while the ML team builds ML products, even though both share common infrastructure for data management. In contrast, Netflix has a core infra team for both data science and ML, and separate algorithm teams. This really shows that there's **no single standardization** across the industry for team structures.

### 3.3. The Iterative Process Stages: A Deep Dive <a name="part-3.3"></a>

Here's a more detailed breakdown of each stage and what happens within them:

1.  **Project Management:**
> - scoping
> - goals
> - objectives
> - constraints
> - budget
> - managing expectations

This is where it all begins! You **plan the budget, manage expectations**, and define the problem you're trying to solve. Chip stresses that an ML system **has to solve a problem**; otherwise, it just creates more maintenance and costs. Don't jump into complex algorithms if a simple heuristic can get you "50% of the way there".

2.  **Data Management:**
> - collect
> - process
> - control
> - store
> - ingest

This stage is all about the data itself. It involves **collecting data, tracking data lineage** (where it comes from), deciding **how to store data** (e.g., data lake vs. data warehouse), and **ingesting** it. This can involve query engines like ElasticSearch or a combination of tools.

3.  **Model Development:**
> - creating datasets
> - labeling
> - feature engineering 
> - model selection 
> - training
> - distributed training
> - evaluation 
> - model optimization 

* Once you have the data, you **create datasets** for training, which includes **labeling, partitioning, sampling, and slicing**.
*  Then comes **model selection, training, and evaluation**. If you're "unlucky," you'll deal with distributed training (which can be "still painful," especially with Python, though companies like Uber and OpenAI are doing great work in this area).
*  Models are then **optimized and compressed** for deployment.

4.  **Deployment**
> - deploying
> - serving
> - detecting and handling data drift
> - releasing strategies 
> - deployment validation 

*   **Deploying** means putting your application on some hardware, while **serving** means responding to user requests. In serverless environments, these terms are often used interchangeably.
*   This stage also involves **initial detection and handling of data drift** (when input data changes over time).
*   You'll define **releasing strategies** and **validate your deployments**.

5.  **Monitoring & Maintenance:**
> - logging
> - tagging
> - tracing 
> - metrics
> - alerts 
> - updates & rollbacks
> - postmortems

*   Chip feels this stage is "still very much underdeveloped for machine learning".
*   However, she believes that **many ML engineering problems can be solved using existing tools from the DevOps world**.
*   **Monitoring is absolutely crucial**. Since you won't have real-time labels for accuracy, you need to monitor **proxies** like conversion rates or click-through rates, but be careful as these can be misleading.
*   Define **metrics for both input data** (e.g., expected ranges, fill rates) and **model output** (e.g., shifts in predicted class distributions). Tools like Grafana or New Relic can help.
*   **Implement tracing with IDs** to track requests through different system components, making debugging much easier.
*   Ensure **feature engineering pipelines for training and production are identical** to avoid subtle but significant bugs. Logging input systems can also be invaluable for future training data.

6.  **Integrating ML into business**
> - business analysis
> - user experience
> - governance 
> - team structure
> - best practices

*   This final stage focuses on **integrating and managing the business impact** of the ML system.
*   It deals with **business analysis, user experience, and governance** (e.g., who has access to what data).

### 3.4. Key Insights on ML Engineering <a name="part-3.4"></a>

Chip also shares some general principles:

*   The tweet that ML engineering is "mostly just machine learning and mostly engineering and not much machine learning" points to the heavy engineering aspect of putting ML into production.
*   ML engineers **should have knowledge about the underlying system and hardware**. Chip references the "hardware lottery," where research ideas gain popularity not because they're inherently better, but because they're better suited to existing hardware. Companies like Apple and Nvidia are working to bridge this gap by developing both chips and applications.

This iterative process highlights that ML in production is a continuous cycle of development, deployment, monitoring, and refinement, requiring a broad set of skills beyond just model building.

## 4. Phases of Adoption <a name="part-4"></a>
She then outlined **four phases of ML adoption** that companies typically go through:

1.  **Before Machine Learning:** Don't jump straight to complex algorithms! Often, a simple heuristic can solve 50% of the problem, giving you a 100% boost. Facebook's news feed, for instance, started chronologically and worked for years before needing complex algorithms.
2.  **Go with Simple ML Models:** Once you decide to use ML, **start simple**. This helps validate your hypothesis, provides visibility into data relationships, and most importantly, **establishes a trustworthy pipeline**. Many outages are caused by engineering issues in distributed pipelines, not ML errors. You need a reliable pipeline before optimizing!
3.  **Optimize Simple Models:** After you have a solid pipeline and validated the approach, optimize your simple models. This can involve feature engineering, collecting more data, or using powerful conventional models like XGBoost or decision trees, which are often surprisingly hard to beat.
4.  **Go with Complex ML Models:** Only after exhausting the potential of simpler models and having a robust pipeline should you consider more complex ML models.

And the overarching principle of good ML system design? It has to **solve a problem**. If it doesn't, it just creates more maintenance and costs.

>**Principles of good ML systems design**
>1. It solves a problem
>2. It's tested
>3. It's accessible to the users
>4. It's ethical
>5. It's components are modular, integrated but separated
>6. It's as simple as possible, but not simpler
>7. It's transparent 
>8. It allows iterative development 
>9. It's versioned
>10. It's documented

### 5. Q&A Highlights: Deeper Dives <a name="part-5"></a>

Here is a more detailed Q&A section from the seminar, drawing on the provided transcript.

***

### Q&A Session with Chip Huyen

**1. Is deployment always easy, or are there settings where it is typically harder?**

Chip Huyen explains that whether deployment is easy or hard **depends heavily on the specific application** and case. She elaborated on the challenges in the medical field:

*   **Practical Challenges**:
    *   **Data Annotation:** Obtaining and labeling patient data is difficult because it requires **domain expertise**, is **slow**, and demands **strict privacy** (often requiring on-site work or extensive training for annotators).
    *   **Data Format:** Medical data comes in many complex formats. While textual data might be easier to handle than image data in general production, medical images (like CT scans) can be **extremely large**, sometimes not even fitting into memory. Data can also be in graph formats (e.g., for drug discovery).
    *   **Time Frame:** Tracking patient data as time series is complex because patient visits are irregular, making it hard to quantify and compare different time intervals (e.g., two years versus two weeks).
*   **Ethical Challenges**:
    *   **Interpretability and Fairness:** These are **extremely important** in medicine because they directly deal with people's lives. Chip highlighted a tweet by Geoffrey Hinton questioning whether an AI surgeon with a 90% cure rate but no explanation is preferable to a doctor with an 80% cure rate who can explain their decisions, emphasizing the ongoing challenge of public trust in AI systems.

**2. How do you deal with concept drift, distributional shift, and updating models in production?**

Chip identifies this as a **major bottleneck** in Machine Learning (ML). She offers several suggestions:

*   **Monitor Models Actively:**
    *   It's **tricky to monitor based on accuracy** in production because labeled data is often unavailable in real-time.
    *   If a system is tied directly to business goals (e.g., recommendation systems), you might get real-time feedback (like click-through rates or conversion rates). However, **proxies can be misleading**; for example, Google's shift from a search engine to a question-answering engine initially caused a drop in click-through rates for advertisers, but ultimately increased revenue because the results were more relevant, leading to higher user satisfaction.
    *   Define **different metrics** for both data and models. For data, you can define expected ranges for input data and monitor when data falls outside these ranges (e.g., a field usually 90% filled now only 50% filled). For models, monitor changes in output distributions (e.g., a class typically predicted 90% of the time now drops to 2%).
    *   Existing tools from the DevOps world (like New Relic) can be adapted for monitoring these metrics.
*   **Implement Tracing:**
    *   In systems with many components, tracing allows you to **track a user's input through different steps** of the system using an ID, which is crucial for debugging when something goes wrong.
    *   This also applies during ML model development, enabling you to step through the system and see the expected output at each stage, even with single inputs.
*   **Ensure Feature Engineering Consistency:**
    *   A common source of bugs arises because **feature engineering pipelines often differ between training and deployment** environments. In training, it's typically done in batch, while in deployment, it's done per sample.
    *   It is critical to **ensure that the feature engineering process yields the exact same output** in both production and training environments. Logging input systems can be very useful for this, and this logged data can even be used as new training data for future model iterations.

**3. How can we encourage ML researchers to prioritize metrics like fairness and interpretability over just top-1 accuracy?**

Chip notes that this is an **active area of work** in the research community. She suggests:

*   **Adaptive Utility Leaderboards:**
    *   Researchers are often motivated by leaderboards and achieving "state-of-the-art" (SOTA) results.
    *   Instead of a single leaderboard focused solely on performance, there should be **adaptive leaderboards where users can specify the importance of different variables** (e.g., performance, interpretability, inference costs).
    *   She mentions work by Cowen and Danjerovsky on "critical NLP leaderboards" that propose using **absolute functions** to create a score based on the importance of different variables.
*   **Benchmarking Beyond Accuracy:**
    *   She highlights benchmarks like **MLPerf**, which focuses on hardware efficiency for both training and inference, as well as dollar costs on cloud providers. These types of benchmarks encourage a broader view of model utility beyond just accuracy.
    *   Chip also mentions ongoing efforts at Facebook (DinoBench) to measure models on different criteria by shipping model weights and outputting various metrics.

**4. Should ML engineers have knowledge of the underlying system, including hardware and software constraints?**

Chip strongly agrees with this point, calling it a **"great point"**.

*   She references the "hardware lottery" paper by Sarah Hooker, which posits that some research ideas become popular not because they are inherently better, but because they are **better suited for existing hardware**.
*   Historically, software and hardware development diverged, leading to algorithms that couldn't effectively utilize hardware or hardware that didn't support algorithms well (e.g., **capsule networks** being a brilliant idea that struggled to gain traction because they couldn't run efficiently on GPUs).
*   Chip emphasizes that **ML engineers should understand the underlying system**. She praises companies that integrate both hardware and software development (like Apple, which builds its own chips for ML and then applications for them, and Nvidia).

**5. Will the course videos for the "Machine Learning Systems Design" course be available online and publicly?**

Chip states that making the course videos available online and publicly would be a **decision discussed by the entire team** involved in the course (including the hosts of the seminar) and with Stanford. As a **"huge believer in open source,"** she expressed her desire to share anything she can online.

> Check out CS 329S: Machine Learning Systems Design 
