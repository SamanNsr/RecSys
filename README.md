# RecSys

This codebase serves as a reference for implementing recommendation systems and is also suitable for research purposes. Its modular design allows you to easily extract the necessary components for your specific needs. With minimal dependencies—primarily essential libraries like Pandas and NumPy, which are included in the dependency manager—you can seamlessly integrate and utilize this code in your projects without hassle. Additionally, there is no need for complex packaging due to the limitations in the ecosystem's package creation and management, making it easier for you to work with the code directly.

## How to use it?
This codebase is structured around three main components: Model, Data, and Evaluation. Based on your data structure, you can select the necessary component from the Data section. Additionally, depending on your specific business needs and the problem you are addressing, you can choose an appropriate model from the Model component. Once you have selected your data and model, you can evaluate their performance using the tools and techniques provided in the Evaluation component, ensuring that your recommendation system meets your desired objectives effectively.

## Models

```mermaid
graph LR
    A["Recommendation Systems"]


    subgraph Basic ["Basic Models"]
        D["Collaborative Filtering"]
        E["Content-Based Filtering"]
    end

    subgraph Advanced ["Advanced Models"]
        H["Context-Aware Models"]
        J["Bandit-Based Models"]
        K["Reinforcement Learning Models"]
    end

    %% Basic Models Breakdown
    D --> D1["Memory-based"]
    D1 --> D1-1["User-based"]
    D1 --> D1-2["Item-based"]
    D --> D2["Model-based"]
    D2 --> D2-1["Supervised"]
    D2 --> D2-2["Unsupervised"]

    H --> H1["Time-based Context"]
    H --> H2["Location-based Context"]


    J --> J1["Epsilon-Greedy"]
    J --> J2["UCB"]
    J --> J3["Thompson Sampling"]

    K --> K1["Q-Learning"]
    K --> K2["SARSA"]
    K --> K3["Policy Gradient"]

    %% Connections
    A --- Basic
    A --- Advanced

    %% Styling
    classDef basic fill:#1f77b4,stroke:#333,stroke-width:2px,color:#fff;  %% Soft blue for basic models
    classDef advanced fill:#2ca02c,stroke:#333,stroke-width:2px,color:#fff;  %% Soft green for advanced models
    classDef node,stroke:#333,stroke-width:2px,color:#000;  %% Light gray for individual nodes

    class A basic;
    class D,E basic;
    class F,G,H,I,J,K advanced;
    class D1,D2,D3,E1,E2 node;
    class F1,F2,F3,G1,G2,G3,H1,H2,I1,I2,J1,J2,J3,K1,K2,K3 node;


```

## Data
