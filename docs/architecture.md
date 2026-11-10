
flowchart TD
    A["Input Dataset<br/>JPEG Images"] --> B["Dataset Validation"]
    B --> C["Parallel Image Loading<br/>ThreadPoolExecutor"]
    C --> D["Retry and Error Handling"]
    D --> E["Batch Processing"]
    E --> F["Image Resizing"]
    F --> G["Grayscale Conversion"]
    G --> H["Gaussian Blur"]
    H --> I["Canny Edge Detection"]
    I --> J["Save Processed Images"]

    D --> K["Error CSV and Logs"]
    E --> L["Performance Monitoring"]
    L --> M["Runtime and Throughput"]
    L --> N["Sampled Memory Usage"]

    J --> O["Benchmark Reports"]
    K --> O
    M --> O
    N --> O
