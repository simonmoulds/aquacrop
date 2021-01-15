Crop Parameters
===============

.. currentmodule:: aquacrop

Configuration
~~~~~~~~~~~~~

.. code-block:: python

    [CROP_PARAMETERS]
    calendar_type = 1
    switch_gdd = true
    gdd_method = 1
    filename = crop_params.nc

Options
~~~~~~~

``calendar_type``
    Set the time units used in the crop development routines:
    1 = Calendar days
    2 = Growing degree days

``switch_gdd``
    Switch to convert inputs supplied in calendar days to growing degree days

``gdd_method``
    If ``switch_gdd = true``, this option specifies the conversion method
    to use

``filename``
    NetCDF file containing crop parameters.

In addition to these options, users may supply any parameter values, or none, in
the configuration file directly (e.g. ``crop_type = 1``). When reading crop parameter values, the program initially checks the configuration file, then the supplied netCDF file. Default values are supplied for all parameters except ``crop_type``, ``planting_date``, and ``harvest_date``.
    
Parameters
~~~~~~~~~~

+--------------+-----------------------------+---------+-----------------------+
| Parameter    | Description                 | Units   |      Default [#]_     |
+==============+=============================+=========+=======================+
| crop_type    | Crop category               | \-      | | 1 = Leafy vegetable |
|              |                             |         | | 2 = Root/tuber      |
|	       |		             |	       | | 3 = Fruit/grain     |
|              |                             |         |                       |
+--------------+-----------------------------+---------+-----------------------+
| planting_date| Planting date of crops in   | dd/mm   | \-                    |
|              | simulation                  |         |                       |
|              |                             |         |                       |
+--------------+-----------------------------+---------+-----------------------+
| harvest_date | Harvest date of crops in    | dd/mm   | \-                    |
|              | simulation                  |         |                       |
|              |                             |         |                       |
+--------------+-----------------------------+---------+-----------------------+
| emergence    | Time from sowing/           | Days/   | \-                    | 
|              | transplanting to emergence  | GDDs    |                       |
|              |                             |         |                       |
+--------------+-----------------------------+---------+-----------------------+
| max_rooting  | Time from sowing/           | Days/   | \-                    | 
|              | transplanting to maximum    | GDDs    |                       |
|              | root development            |         |                       |
|              |                             |         |                       |
+--------------+-----------------------------+---------+-----------------------+
| senescence   | Time from sowing/           | Days/   | \-                    | 
|              | transplanting to start of   | GDDs    |                       |
|              | canopy senescence           |         |                       |
|              |                             |         |                       |
+--------------+-----------------------------+---------+-----------------------+
| maturity     | Time from sowing/           | Days/   | \-                    | 
|              | transplanting to crop       | GDDs    |                       |
|              | maturity                    |         |                       |
|              |                             |         |                       |
+--------------+-----------------------------+---------+-----------------------+
| hi_start     | Time from sowing/           | Days/   | \-                    | 
|              | transplanting to start of   | GDDs    |                       |
|              | yield formation             |         |                       |
|              |                             |         |                       |
+--------------+-----------------------------+---------+-----------------------+
| flowering    | Duration of flowering (only | Days/   | \-                    |
|              | relevant for fruit/grain    | GDDs    |                       |
|              | crops)                      |         |                       |
|              |                             |         |                       |
+--------------+-----------------------------+---------+-----------------------+
| yld_form     | Duration of yield formation | Days/   | \-                    | 
|              | period                      | GDDs    |                       |
|              |                             |         |                       |
+--------------+-----------------------------+---------+-----------------------+
| t_base       | Lower temperature limit on  | deg C   |                       | 
|              | crop growth                 |         |                       |
|              |                             |         |                       |
+--------------+-----------------------------+---------+-----------------------+
| t_upp        | Upper temperature limit on  | deg C   |                       |
|              | crop growth                 |         |                       |
+--------------+-----------------------------+---------+-----------------------+
| pol_heat_str | Switch determining whether  | \-      |                       |
|              | pollination is affected     |         |                       |
|              | by heat stress              |         |                       |
|              |                             |         |                       |
+--------------+-----------------------------+---------+-----------------------+
| tmax_up      | Temperature at which        |         |                       |
|              | pollination starts to fail  |         |                       |
|              |                             |         |                       |
+--------------+-----------------------------+---------+-----------------------+
| tmax_lo      | Temperature at which        |         |                       |
|              | pollination fails entirely  |         |                       |
|              |                             |         |                       |
+--------------+-----------------------------+---------+-----------------------+
| pol_cold_str | Switch determining whether  | \-      |                       |
|              | pollination is affected by  |         |                       |
|              | cold stress                 |         |                       |
|              |                             |         |                       |
+--------------+-----------------------------+---------+-----------------------+
| tmin_up      | Temperature below which     | deg C   |                       |
|              | pollination starts to fail  |         |                       |
|              |                             |         |                       |
+--------------+-----------------------------+---------+-----------------------+
| tmin_lo      | Temperature below which     | deg C   |                       |
|              | pollination fails entirely  |         |                       |
|              |                             |         |                       |
+--------------+-----------------------------+---------+-----------------------+
| bio_heat_str | Switch determining whether  | \-      |                       |
|              | biomass production is       |         |                       |
|              | affected by heat stress     |         |                       |
|              |                             |         |                       |
+--------------+-----------------------------+---------+-----------------------+
| gdd_up       | Minimum GDDs required for   | GDDs    |                       |
|              | full biomass production     |         |                       |
|              |                             |         |                       |
+--------------+-----------------------------+---------+-----------------------+
| gdd_lo       | Minimum GDDs required for   | GDDs    |                       |
|              | any amount of biomass       |         |                       |
|              | production to occur         |         |                       |
|              |                             |         |                       |
+--------------+-----------------------------+---------+-----------------------+
| fshape_b     | Shape factor controlling    |         |                       |
|              | the reduction in biomass    |         |                       |
|              | production as a result of   |         |                       |
|              | accumulating insufficient   |         |                       |
|              | GDDs                        |         |                       |
|              |                             |         |                       |
+--------------+-----------------------------+---------+-----------------------+
| pct_zmin     | Fraction of minimum         |         | 0.7                   |
|              | effective root depth at     |         |                       |
|              | sowing/transplanting        |         |                       |
|              |                             |         |                       |
+--------------+-----------------------------+---------+-----------------------+
| z_min        | Minimum effective root      | m       |                       |
|              | depth                       |         |                       |
|              |                             |         |                       |
+--------------+-----------------------------+---------+-----------------------+
| z_max        | Maximum effective root      | m       |                       |
|              | depth                       |         |                       |
|              |                             |         |                       |
+--------------+-----------------------------+---------+-----------------------+
| fshape_r     | Shape factor controlling    | \-      | 1.5                   |
|              | the decrease in the rate of |         |                       |
|              | root expansion over time    |         |                       |
|              |                             |         |                       |
+--------------+-----------------------------+---------+-----------------------+
| fshape_ex    | Shape factor controlling    | \-      | -6                    |
|              | the effect of water stress  |         |                       |
|              | on root expansion           |         |                       |
|              |                             |         |                       |
+--------------+-----------------------------+---------+-----------------------+
| sx_topq      | Maximum water extraction at | m3 m-3  |                       | 
|              | the top of the root zone    | day-1   |                       |
|              |                             |         |                       |
+--------------+-----------------------------+---------+-----------------------+
| sx_botq      | Maximum water extraction at | m3 m-3  |                       |
|              | the bottom of the root zone | day-1   |                       |
|              |                             |         |                       |
+--------------+-----------------------------+---------+-----------------------+
| a_tr         | Exponent parameter          | \-      | 1                     |
|              | controlling the reduction   |         |                       |
|              | in transpiration and        |         |                       |
|              | photosynthetic capacity as  |         |                       |         
|              | a result of canopy decline  |         |                       |
|              |                             |         |                       |
+--------------+-----------------------------+---------+-----------------------+
| seed_size    | The surface area of soil    | cm2     |                       |
|              | covered by an individual    |         |                       |
|              | seedling at 90% emergence   |         |                       |
|              |                             |         |                       |
+--------------+-----------------------------+---------+-----------------------+
| plant_pop    | Area density of plants      | plants  |                       |
|              |                             | ha-1    |                       |
+--------------+-----------------------------+---------+-----------------------+
| cc_min       | Canopy cover below          | \-      |                       |
|              | which yield formation does  |         |                       |
|              | not occur                   |         |                       | 
|              |                             |         |                       |
+--------------+-----------------------------+---------+-----------------------+
| cc_x         | Maximum canopy cover        | \-      |                       |
|              |                             |         |                       |
+--------------+-----------------------------+---------+-----------------------+
| cdc          | Canopy decline coefficient  | day-1   |                       |
|              |                             | GDD-1   |                       |
|              |                             |         |                       |
+--------------+-----------------------------+---------+-----------------------+
| cgc          | Canopy growth coefficient   | day-1   |                       |
|              |                             | GDD-1   |                       |
|              |                             |         |                       |
+--------------+-----------------------------+---------+-----------------------+
| kcb          | Maximum crop coefficient    | \-      |                       |
|              | corresponding to fully      |         |                       |
|              | developed canopy            |         |                       |
|              |                             |         |                       |
+--------------+-----------------------------+---------+-----------------------+
| f_age        | Decline of crop coefficient | % day-1 |                       |
|              | with canopy cover age       |         |                       |
|              |                             |         |                       |
+--------------+-----------------------------+---------+-----------------------+
| wp           | Water productivity,         | g m-2   |                       |
|              | normalised for ETo and      |         |                       |
|              | atmospheric carbon dioxide  |         |                       |
|              | concentration               |         |                       |
|              |                             |         |                       |
+--------------+-----------------------------+---------+-----------------------+
| wpy          | Amount to adjust water      | %       |                       |
|              | productivity during yield   |         |                       |
|              | formation stage             |         |                       |
|              |                             |         |                       |
+--------------+-----------------------------+---------+-----------------------+
| f_sink       | Coefficient of crop sink    | \-      |                       |
|              | strength                    |         |                       |
|              |                             |         |                       |
+--------------+-----------------------------+---------+-----------------------+
| b_sted       | Parameter for adjusting     | \-      |                       |
|              | water productivity to       |         |                       |
|              | account for the effects of  |         |                       |
|              | carbon dioxide              |         |                       |
|              | (Steduto et al., 2007)      |         |                       |
|              |                             |         |                       |
+--------------+-----------------------------+---------+-----------------------+
| b_face       | As above, except values are | \-      |                       |
|              | those derived from the      |         |                       |
|              | FACE experiments            |         |                       |
|              |                             |         |                       |
+--------------+-----------------------------+---------+-----------------------+
| hi_0         | Reference harvest index     | \-      |                       |
|              |                             |         |                       |
+--------------+-----------------------------+---------+-----------------------+
| hi_ini       | Initial harvest index       | \-      |                       |
|              |                             |         |                       |
+--------------+-----------------------------+---------+-----------------------+
| dhi_pre      | Increase in harvest index   | %       |                       |
|              | arising from pre-anthesis   |         |                       |
|              | water stress                |         |                       |
|              |                             |         |                       |
+--------------+-----------------------------+---------+-----------------------+
| a_hi         | Coefficient to relate the   | \-      |                       |
|              | positive impact of          |         |                       |
|              | restricted post-anthesis    |         |                       |
|              | vegetative growth on        |         |                       |
|              | harvest index               |         |                       |
|              |                             |         |                       |
+--------------+-----------------------------+---------+-----------------------+
| b_hi         | Coefficient to relate the   | \-      |                       |
|              | negative impact of          |         |                       |
|              | restricted post-anthesis    |         |                       |
|              | vegetative growth on        |         |                       |
|              | harvest index               |         |                       |
|              |                             |         |                       |
+--------------+-----------------------------+---------+-----------------------+
| dhi_0        | Maximum increase in harvest | %       |                       |
|              | index above the reference   |         |                       |
|              | value                       |         |                       |
|              |                             |         |                       |
+--------------+-----------------------------+---------+-----------------------+
| determinant  | Switch indicating whether   | \-      |                       |
|              | the crop is determinate     |         |                       |
|              |                             |         |                       |
+--------------+-----------------------------+---------+-----------------------+
| exc          | Excess of potential fruits  | %       |                       |
|              |                             |         |                       |
+--------------+-----------------------------+---------+-----------------------+
| max_flow_pct | Percentage of total         | %       |                       |
|              | flowering period at which   |         |                       |
|              | maximum flowering takes     |         |                       |
|              | place                       |         |                       |
|              |                             |         |                       |
+--------------+-----------------------------+---------+-----------------------+
| p_up1        | Upper soil water depletion  |         |                       |
|              | threshold for water stress  |         |                       |
|              | effects on canopy expansion |         |                       |
|              |                             |         |                       |
+--------------+-----------------------------+---------+-----------------------+
| p_up2        | Upper soil water depletion  |         |                       |
|              | threshold for water stress  |         |                       |
|              | effects on stomatal control |         |                       |
|              |                             |         |                       |
+--------------+-----------------------------+---------+-----------------------+
| p_up3        | Upper soil water depletion  |         |                       |
|              | threshold for water stress  |         |                       |
|              | effects on canopy           |         |                       |
|              | senescence                  |         |                       |
|              |                             |         |                       |
+--------------+-----------------------------+---------+-----------------------+
| p_up4        | Upper soil water depletion  |         |                       |
|              | threshold for water stress  |         |                       |
|              | effects on crop pollination |         |                       |
|              |                             |         |                       |
+--------------+-----------------------------+---------+-----------------------+
| p_lo1        | Lower soil water depletion  |         |                       |
|              | threshold for water stress  |         |                       |
|              | effects on canopy expansion |         |                       |
|              |                             |         |                       |
+--------------+-----------------------------+---------+-----------------------+
| p_lo2        | Upper soil water depletion  |         |                       |
|              | threshold for water stress  |         |                       |
|              | effects on stomatal control |         |                       |
|              |                             |         |                       |
+--------------+-----------------------------+---------+-----------------------+
| p_lo3        | Upper soil water depletion  |         |                       |
|              | threshold for water stress  |         |                       |
|              | effects on canopy           |         |                       |
|              | senescence                  |         |                       |
|              |                             |         |                       |
+--------------+-----------------------------+---------+-----------------------+
| p_lo4        | Upper soil water depletion  |         |                       |
|              | threshold for water stress  |         |                       |
|              | effects on crop pollination |         |                       |
|              |                             |         |                       |
+--------------+-----------------------------+---------+-----------------------+
| fshape_w1    | Shape factor controlling    |         |                       |
|              | the effect of water stress  |         |                       |
|              | on canopy expansion         |         |                       |
|              |                             |         |                       |
+--------------+-----------------------------+---------+-----------------------+
| fshape_w2    | Shape factor controlling    |         |                       |
|              | the effect of water stress  |         |                       |
|              | on stomatal control         |         |                       |
|              |                             |         |                       |
+--------------+-----------------------------+---------+-----------------------+
| fshape_w3    | Shape factor controlling    |         |                       |
|              | the effect of water stress  |         |                       |
|              | on canopy senescence        |         |                       |
|              |                             |         |                       |
+--------------+-----------------------------+---------+-----------------------+
| fshape_w4    | Shape factor controlling    |         |                       |
|              | the effect of water stress  |         |                       |
|              | on crop pollination         |         |                       |
|              |                             |         |                       |
+--------------+-----------------------------+---------+-----------------------+
| et_adj       | Switch to adjust water      |         |                       |
|              | stress thresholds for daily |         |                       |
|              | variations in ETo           |         |                       |
|              |                             |         |                       |
+--------------+-----------------------------+---------+-----------------------+
| aer          | Water depletion at which    |         |                       |
|              | aeration stress begins to   |         |                       |
|              | occur                       |         |                       |
|              |                             |         |                       |
+--------------+-----------------------------+---------+-----------------------+
| lag_aer      | Time lag in before aeration |         |                       |
|              | stress affects crop         |         |                       |
|              | development                 |         |                       |
|              |                             |         |                       |
+--------------+-----------------------------+---------+-----------------------+
| beta         | Reduction to ``p_lo3``      | %       | 12                    |
|              | as a result of early canopy |         |                       |
|              | senescence due to water     |         |                       |
|              | stress                      |         |                       |
|              |                             |         |                       |
+--------------+-----------------------------+---------+-----------------------+
| germ_thr     | Fraction of total available | \-      | 0.20                  |
|              | water in the root zone      |         |                       |
|              | which is needed for         |         |                       |
|              | germination                 |         |                       |
|              |                             |         |                       |
+--------------+-----------------------------+---------+-----------------------+

.. [#] Only parameters with generic (not crop-specific) default values are shown here.
