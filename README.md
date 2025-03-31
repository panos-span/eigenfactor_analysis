## Implementation and analysis of the EigenFactor-adjusted H-Index. 
The EigenFactor metric evaluates the quality of journals based on their influence and impact. The implementation includes:

- Data extraction from the Alexandria3k tool.
- Calculation of the EigenFactor-adjusted H-Index for top authors.
- Analysis of citation practices and patterns based on the EigenFactor ranking.
- Comparison of top authors publishing in lower-tier journals with random authors publishing in top-tier journals.

## Requirements

For this project, we will use the following libraries:
- `alexandria3k` for data extraction.
- `rdbunit` for database unit testing.

And more common datatools like `pandas`, `numpy`, `matplotlib`, and `seaborn`.

Exact requirements can be found in the `requirements.txt` file.

## Instructions

### Prerequisites

Files needed externally for the project:

- ``get_citation_network.txt``: This file is needed to get the citation network from the Alexandria3k tool. It is a copy of the original file, which is not included in this repository. You can find the necessary tables for it's generation in the `citation_network_(if)` folder.
- ``get_issn_subject``: This file is needed to get the ISSN and subject of the journals from the Alexandria3k tool. It is a copy of the original file, which is not included in this repository. In order to get the ISSN and subject of the journals, you need to run a SQL query on the Alexandria3k database on the Crossref 2023 database.

### Install Crossref 202X

Install at the desired path with the following command:

```bash
aria2c http://dx.doi.org/XXX/XXX &&
```

Then for convenience, rename the directory`:

```bash
mv 'April 2024 Public Data File from Crossref' Crossref-April-2024
```

### Populate Database

Now we can populate the database with the Crossref data. In our folder we can now run the following command:

```bash
make populate
```

> Warning: This will take a while, as it will download the entire Crossref database and populate the database with it. 

> Additional Note: According to the year of the Crossref data, the database will be populated with the data from the year 202X.
> The years set in the `Makefile` and can be changed if needed.


## Structure 

The project is structured as follows:

### Data Processing
* **citation_network_(if)/** - Citation network extraction code for eigenfactor calculation
* **common/** - Makefiles for database population and dependency installation
* **base_tables/** - Core tables that form the backbone of the analysis structure
* **top_tables/** - Top tables generation for eigenfactor analysis
* **bottom_tables/** - Bottom tables generation for eigenfactor analysis

### Analysis Components
* **simple_citation_analysis/** - Simple citation analysis for eigenfactor-adjusted h-index
* **citation_network_analysis/** - Complex citation network analysis for top/bottom authors
* **analysis_files/** - General analysis files for eigenfactor tables
* **orcid_h5_calculations/** - Base calculations for h5-index

### Comparison tables & Statistics
* **mw_statistics/** - Mann-Whitney U statistics analysis
* **random_authors_tables/** - Random author tables for citation network comparison
* **author_matching/** - Author matching between top and bottom tables

### Output & Testing
* **print/** - Code for table printing and basic statistics
* **tests/** - Test suite using `rdbunit` for code validation