## 具体的过程
是选择BM和市值两个公司指标，并用它们进行了  $2\times 3$  独立双重排序，在排序时，以深交所上市公司的市值中位数为界将上市公司上市公司BM分成小市值（Small）和大市值（Big）两组。类似的，以深交所  $30\%$  和  $70\%$  分位数为界，BM高于  $70\%$  分位数的为High组、BM低于  $30\%$  分位数的为Low组、位于中间的为Middle组。通过以上划分后，按照市值和BM各自所属的组别，所有股票被分到一共6（  $2\times 3 = 6$  ）个组中，记为S/H、S/M、S/L、B/H、B/M及B/L。将每组中的股票收益率按市值加权就得到六个投资组合。最终，Fama and French（1993）使用如下方法构建了规模和价值两个因子：

$$
\begin{array}{l}\mathrm{SMB} = \frac{1}{3} (\mathrm{S / H} + \mathrm{S / M} + \mathrm{S / L}) - \frac{1}{3} (\mathrm{B / H} + \mathrm{B / M} + \mathrm{B / L})\\ \mathrm{HML} = \frac{1}{2} (\mathrm{S / H} + \mathrm{B / H}) - \frac{1}{2} (\mathrm{S / L} + \mathrm{B / L}) \end{array} \tag{4.2}
$$