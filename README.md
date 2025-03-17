<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
<div align="left">

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]

</div>

<a href="https://github.com/Kaweees/trainer">
  <img alt="Python Logo" src="assets/img/python.png" align="right" width="150">
</a>

<div align="left">
  <h1><em><a href="https://miguelvf.dev/blog/dotfiles/compendium">~trainer</a></em></h1>
</div>

<!-- ABOUT THE PROJECT -->

A web dashboard for training machine learning models with PyTorch.

### Built With

[![Python][Python-shield]][Python-url]
[![Pytest][Pytest-shield]][Pytest-url]
[![Codecov][Codecov-shield]][Codecov-url]
[![Docker][Docker-shield]][Docker-url]
[![GitHub Actions][github-actions-shield]][github-actions-url]

<!-- GETTING STARTED -->

## Getting Started

### Prerequisites

Before attempting to build this project, make sure you have [Python 3.10 or higher](https://www.python.org/downloads/), [just](https://just.systems/), and [uv](https://docs.astral.sh/uv/#getting-started) installed on your machine.

### Installation

To get a local copy of the project up and running on your machine, follow these simple steps:

1. Clone the project repository

   ```sh
   git clone https://github.com/Kaweees/trainer.git
   cd trainer
   ```

2. Install the virtual environment and pre-commit hooks

   ```sh
   just install
   ```

3. Set up environment variables

   ```sh
   cp .env.example .env
   ```

   Edit the `.env` file values to reflect your environment.

4. Run the project

   ```sh
   just run <package_name>
   ```

<!-- PROJECT FILE STRUCTURE -->

## Project Structure

```sh
trainer/
├── .github/                       - GitHub Actions CI/CD workflows
├── scripts/                       - Standalone scripts
├── shared/
│   └── utils/                     - Shared utility functions
├── src/                           - Project packages
│   ├── core/                      - Core application logic
│   └── ...                        - Other packages
├── tests/                         - Project tests (mirrors the main project structure)
├── .env.example                   - Reference environment variables file
├── LICENSE                        - Project license
└── README.md                      - You are here
```

## License

The source code for this project is distributed under the terms of the MIT License, as I firmly believe that collaborating on free and open-source software fosters innovations that mutually and equitably beneficial to both collaborators and users alike. See [`LICENSE`](./LICENSE) for details and more information.

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[contributors-shield]: https://img.shields.io/github/contributors/Kaweees/trainer.svg?style=for-the-badge
[contributors-url]: https://github.com/Kaweees/trainer/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/Kaweees/trainer.svg?style=for-the-badge
[forks-url]: https://github.com/Kaweees/trainer/network/members
[stars-shield]: https://img.shields.io/github/stars/Kaweees/trainer.svg?style=for-the-badge
[stars-url]: https://github.com/Kaweees/trainer/stargazers

<!-- MARKDOWN SHIELD BAGDES & LINKS -->
<!-- https://github.com/Ileriayo/markdown-badges -->

[Python-shield]: https://img.shields.io/badge/Python-%23008080.svg?style=for-the-badge&logo=python&logoColor=FFDD54&labelColor=222222&color=306998
[Python-url]: https://www.python.org/
[Pytest-shield]: https://img.shields.io/badge/pytest-%23008080.svg?style=for-the-badge&logo=pytest&logoColor=2F9FE3&labelColor=222222&color=2F9FE3
[Pytest-url]: https://docs.pytest.org
[Docker-shield]: https://img.shields.io/badge/docker-%232671E5.svg?style=for-the-badge&logo=docker&logoColor=1D63ED&labelColor=222222&color=1D63ED
[Docker-url]: https://www.docker.com/
[Codecov-shield]: https://img.shields.io/badge/codecov-%23008080.svg?style=for-the-badge&logo=codecov&logoColor=FF0077&labelColor=222222&color=FF0077
[Codecov-url]: https://codecov.io/
[github-actions-shield]: https://img.shields.io/badge/github%20actions-%232671E5.svg?style=for-the-badge&logo=githubactions&logoColor=2671E5&labelColor=222222&color=2671E5
[github-actions-url]: https://github.com/features/actions
