# Introduction

This is a simple portfolio chooser based on time series of investment funds, based
on the [Sharpe Ratio](https://en.wikipedia.org/wiki/Sharpe_ratio) and the
[Efficient Frontier](https://en.wikipedia.org/wiki/Efficient_frontier).

## Configuration

Main parameters for the run of the portfolio chooser can be selected in the
`config/config.toml` file.

## Running

You can directly run the full pipeline with

```bash
make run
```

or directly

```bash
python -m investments.main
```

and check the visualization of the efficient frontier with

```bash
make viz
```

or

```bash
make viz_hull
```

In order to run the visualization commands, you need Firefox. In case you don't have
it, just open the corresponding htmls in the data folder directly instead.

## Pipeline

Each part of the pipeline can be run separately through the according subpackage.

### Raw files

For the fund time series, we are currently capturing monthly data by using
<https://dados.cvm.gov.br/dataset/fii-doc-inf_mensal>. This seems to be restricted
only to real estate, so we probably want to expand this in the future.

For the CDI time series, we capture the data directly by copy-pasting the data
in the following link: <https://brasilindicadores.com.br/cdi/>.

### Preprocessed files

Preprocessing transforms the rentability into a simple multiplier, e.g. a monthly
rentability of `+1.2%` gets translated into `1.012` on a `"values"` column.

A `"dt"` column contains date in the format `YYYY-MM-01`. The funds-related csv,
`"funds.csv"` also has an additional column `"CNPJ_Fundo"`, corresponding to an
identifier of the fund (c.f. <https://www.gov.br/receitafederal/pt-br/servicos/cadastro/cnpj>).

To run this part of the pipeline, run

```bash
python -m investments.preprocess.main
```

### Models

Consists of pickled files of the time series of each fund, according to the `TimeSeries`
class.

To run this part of the pipeline, run

```bash
python -m investments.models.main
```

### Outputs

We get the risk-return plots, both in `.png` and `.html` format, the later for some
interactiveness.

Furthermore, we get the convex hull of the plot to more easily identify the efficient
frontier.

To run this part of the pipeline, run

```bash
python -m investments.outputs.main
```
