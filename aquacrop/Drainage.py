#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from numba import jit

import aquacrop_fc

class Drainage(object):
    """Class to infiltrate incoming water"""
    
    def __init__(self, Drainage_variable):
        self.var = Drainage_variable

    def initial(self):
        self.var.FluxOut = np.zeros((self.var.nFarm, self.var.nCrop, self.var.nComp, self.var.nCell))
        self.var.DeepPerc = np.zeros((self.var.nFarm, self.var.nCrop, self.var.nCell))
        self.var.Recharge = np.zeros((self.var.nCell))

        # experiment
        # self.var.th_exc_comp = self.var.th_sat_comp - self.var.th_fc_comp
        # self.var.exp_th_exc_comp = np.exp(self.var.th_exc_comp)
        
    # TODO: install numba and try this
    @staticmethod
    @jit(nopython=True, cache=True)
    def compute_dthdt(th, th_sat, th_fc, th_fc_adj, tau):
        cond1 = th <= th_fc_adj
        dthdt1 = np.zeros_like(th, dtype=np.float64)
        dthdt = (dthdt1 * cond1)
        cond2 = np.logical_not(cond1) & (th >= th_sat)
        dthdt2 = tau * (th_sat - th_fc)
        dthdt += (dthdt2 * cond2)
        cond3 = np.logical_not(cond1 | cond2)
        dthdt3 = (tau * (th_sat - th_fc) * ((np.exp(th - th_fc) - 1) / (np.exp(th_sat - th_fc) - 1)))
        dthdt += (dthdt3 * cond3)
        cond4 = (cond2 | cond3) & ((th - dthdt) < th_fc_adj)
        dthdt4 = (th - th_fc_adj)
        dthdt += (dthdt4 * cond4)
        return dthdt
        
    def dynamic(self):
        """Function to redistribute stored soil water"""
        # dims = self.var.th.shape
        thnew = np.copy(self.var.th)
        drainsum = np.zeros((self.var.nFarm, self.var.nCrop, self.var.nCell))
        for comp in range(self.var.nComp):

            # Calculate drainage ability of compartment ii
            dthdt = self.compute_dthdt(
                self.var.th[:,:,comp,:],
                self.var.th_sat_comp[:,:,comp,:],
                self.var.th_fc_comp[:,:,comp,:],
                self.var.th_fc_adj[:,:,comp,:],
                self.var.tau_comp[:,:,comp,:]
            )

            # Drainage from compartment ii (mm) (Line 41 in AOS_Drainage.m)
            draincomp = dthdt * self.var.dz[comp] * 1000

            # Check drainage ability of compartment ii against cumulative
            # drainage from compartments above (Lines 45-52 in AOS_Drainage.m)
            excess = np.zeros((self.var.nFarm, self.var.nCrop, self.var.nCell))
            prethick = self.var.dz_sum[comp] - self.var.dz[comp]
            drainmax = dthdt * 1000 * prethick
            drainability = (drainsum <= drainmax)

            # Drain compartment
            cond5 = drainability
            thnew[:,:,comp,:][cond5] = (self.var.th[:,:,comp,:] - dthdt)[cond5]

            # Update cumulative drainage (mm), restrict to saturated hydraulic
            # conductivity and adjust excess drainage flow
            drainsum[cond5] += draincomp[cond5]
            cond51 = (cond5 & (drainsum > self.var.k_sat_comp[:,:,comp,:]))
            excess[cond51] += (drainsum - self.var.k_sat_comp[:,:,comp,:])[cond51]
            drainsum[cond51] = self.var.k_sat_comp[:,:,comp,:][cond51]

            # Calculate value of theta (thX) needed to provide a drainage
            # ability equal to cumulative drainage (Lines 70-85 in AOS_Drainage.m)

            # TODO: put this in a staticmethod, use numba            
            cond6 = np.logical_not(drainability)
            dthdt[cond6] = np.divide(drainsum, 1000 * prethick, out=np.zeros_like(drainsum), where=prethick!=0)[cond6]
            thX = np.zeros((self.var.nFarm, self.var.nCrop, self.var.nCell))
            cond61 = (cond6 & (dthdt <= 0))
            thX[cond61] = self.var.th_fc_adj[:,:,comp,:][cond61]
            cond62 = (cond6 & np.logical_not(cond61) & (self.var.tau_comp[:,:,comp,:] > 0))
            A = (1 + ((dthdt * (np.exp(self.var.th_sat_comp[:,:,comp,:] - self.var.th_fc_comp[:,:,comp,:]) - 1)) / (self.var.tau_comp[:,:,comp,:] * (self.var.th_sat_comp[:,:,comp,:] - self.var.th_fc_comp[:,:,comp,:]))))
            thX[cond62] = (self.var.th_fc_adj[:,:,comp,:] + np.log(A))[cond62]
            thX[cond62] = np.clip(thX, self.var.th_fc_adj[:,:,comp,:], None)[cond62]
            cond63 = (cond6 & np.logical_not(cond61 | cond62))
            thX[cond63] = (self.var.th_sat_comp[:,:,comp,:] + 0.01)[cond63]

            # Check thX against hydraulic properties of current soil layer

            # Increase compartment ii water content with cumulative drainage
            cond64 = (cond6 & (thX <= self.var.th_sat_comp[:,:,comp,:]))
            thnew[:,:,comp,:][cond64] = (self.var.th[:,:,comp,:] + (drainsum / (1000 * self.var.dz[comp])))[cond64]

            # Cumulative drainage is the drainage difference between theta_x and
            # new theta plus drainage ability at theta_x
            cond641 = (cond64 & (thnew[:,:,comp,:] > thX))
            drainsum[cond641] = ((thnew[:,:,comp,:] - thX) * 1000 * self.var.dz[comp])[cond641]

            # Calculate drainage ability for thX
            dthdt = self.compute_dthdt(
                thX,
                self.var.th_sat_comp[:,:,comp,:],
                self.var.th_fc_comp[:,:,comp,:],
                self.var.th_fc_adj[:,:,comp,:],
                self.var.tau_comp[:,:,comp,:]
            )

            # Update cumulative drainage (mm), restrict to saturated hydraulic
            # conductivity and adjust excess drainage flow
            drainsum[cond641] += (dthdt * 1000 * self.var.dz[comp])[cond641]
            cond6415 = (cond641 & (drainsum > self.var.k_sat_comp[:,:,comp,:]))
            excess[cond6415] += (drainsum - self.var.k_sat_comp[:,:,comp,:])[cond6415]
            drainsum[cond6415] = self.var.k_sat_comp[:,:,comp,:][cond6415]

            # Update water content
            thnew[:,:,comp,:][cond641] = (thX - dthdt)[cond641]

            # Calculate drainage ability for updated water content
            cond642 = (cond64 & np.logical_not(cond641) & (thnew[:,:,comp,:] > self.var.th_fc_adj[:,:,comp,:]))
            dthdt = self.compute_dthdt(
                thnew[:,:,comp,:],
                self.var.th_sat_comp[:,:,comp,:],
                self.var.th_fc_comp[:,:,comp,:],
                self.var.th_fc_adj[:,:,comp,:],
                self.var.tau_comp[:,:,comp,:]
            )

            # Update water content
            thnew[:,:,comp,:][cond642] = (thnew[:,:,comp,:] - dthdt)[cond642]

            # Update cumulative drainage (mm), restrict to saturated hydraulic
            # conductivity and adjust excess drainage flow
            drainsum[cond642] = (dthdt * 1000 * self.var.dz[comp])[cond642]
            cond6425 = (cond642 & (drainsum > self.var.k_sat_comp[:,:,comp,:]))
            excess[cond6425] += (drainsum - self.var.k_sat_comp[:,:,comp,:])[cond6425]
            drainsum[cond6425] = self.var.k_sat_comp[:,:,comp,:][cond6425]

            # Otherwise, drainage is zero
            cond643 = (cond64 & np.logical_not(cond641 | cond642))
            drainsum[cond643] = 0

            # Increase water content in compartment ii with cumulative
            # drainage from above
            cond65 = (cond6 & np.logical_not(cond64) & (thX > self.var.th_sat_comp[:,:,comp,:]))
            thnew[:,:,comp,:][cond65] = (self.var.th[:,:,comp,:] + (drainsum / (1000 * self.var.dz[comp])))[cond65]

            # Check new water content against hydraulic properties of soil layer
            # Lines 166-198
            cond651 = (cond65 & (thnew[:,:,comp,:] <= self.var.th_sat_comp[:,:,comp,:]))

            # Calculate new drainage ability
            cond6511 = (cond651 & (thnew[:,:,comp,:] > self.var.th_fc_adj[:,:,comp,:]))
            dthdt = self.compute_dthdt(
                thnew[:,:,comp,:],
                self.var.th_sat_comp[:,:,comp,:],
                self.var.th_fc_comp[:,:,comp,:],
                self.var.th_fc_adj[:,:,comp,:],
                self.var.tau_comp[:,:,comp,:]
            )

            # Update water content
            thnew[:,:,comp,:][cond6511] -= (dthdt)[cond6511]

            # Update cumulative drainage (mm), restrict to saturated hydraulic
            # conductivity and adjust excess drainage flow
            drainsum[cond6511] = (dthdt * 1000 * self.var.dz[comp])[cond6511]
            cond65115 = (cond6511 & (drainsum > self.var.k_sat_comp[:,:,comp,:]))
            excess[cond65115] += (drainsum - self.var.k_sat_comp[:,:,comp,:])[cond65115]
            drainsum[cond65115] = self.var.k_sat_comp[:,:,comp,:][cond65115]

            cond6512 = (cond651 & (np.logical_not(cond6511)))
            drainsum[cond6512] = 0

            # Calculate excess drainage above saturation
            cond652 = (cond65 & np.logical_not(cond651) & (thnew[:,:,comp,:] > self.var.th_sat_comp[:,:,comp,:]))
            excess[cond652] = ((thnew[:,:,comp,:] - self.var.th_sat_comp[:,:,comp,:]) * 1000 * self.var.dz[comp])[cond652]

            # Calculate drainage ability for updated water content
            dthdt = self.compute_dthdt(
                thnew[:,:,comp,:],
                self.var.th_sat_comp[:,:,comp,:],
                self.var.th_fc_comp[:,:,comp,:],
                self.var.th_fc_adj[:,:,comp,:],
                self.var.tau_comp[:,:,comp,:]
            )

            # Update water content
            thnew[:,:,comp,:][cond652] = (self.var.th_sat_comp[:,:,comp,:] - dthdt)[cond652]

            # Update drainage, maximum drainage, excess drainage
            draincomp[cond652] = (dthdt * 1000 * self.var.dz[comp])[cond652]
            drainmax[cond652] = (dthdt * 1000 * prethick)[cond652]
            drainmax[cond652] = np.clip(drainmax, None, excess)[cond652]
            excess[cond652] -= drainmax[cond652]

            # Update cumulative drainage (mm), restrict to saturated hydraulic
            # conductivity and adjust excess drainage flow
            drainsum[cond652] = (draincomp + drainmax)[cond652]
            cond6525 = (cond652 & (drainsum > self.var.k_sat_comp[:,:,comp,:]))
            excess[cond6525] += (drainsum - self.var.k_sat_comp[:,:,comp,:])[cond6525]
            drainsum[cond6525] = self.var.k_sat_comp[:,:,comp,:][cond6525]

            # Store output flux from compartment ii
            self.var.FluxOut[:,:,comp,:] = np.copy(drainsum)

            # TODO: make a staticmethod for this loop?
            
            # Redistribute excess in compartment above
            precomp = comp + 1
            while (np.any(excess > 0)) & (precomp != 0):

                # Include condition here so that it is updated
                cond7 = (excess > 0)

                # Update compartment counter
                precomp -= 1

                # Update flux from compartment
                if (precomp < comp):
                    self.var.FluxOut[:,:,precomp,:][cond7] -= excess[cond7]

                # Increase water content to store excess
                thnew[:,:,precomp,:][cond7] += (excess / (1000 * self.var.dz[precomp]))[cond7]

                # Limit water content to saturation and adjust excess counter
                cond71 = (cond7 & (thnew[:,:,precomp,:] > self.var.th_sat_comp[:,:,precomp,:]))
                excess[cond71] = ((thnew[:,:,precomp,:] - self.var.th_sat_comp[:,:,precomp,:]) * 1000 * self.var.dz[precomp])[cond71]
                thnew[:,:,precomp,:][cond71] = self.var.th_sat_comp[:,:,precomp,:][cond71]

                cond72 = (cond7 & np.logical_not(cond71))
                excess[cond72] = 0

        self.var.DeepPerc = np.copy(drainsum)
        self.var.Recharge = np.sum(
            np.multiply(
                (drainsum / 1000.),
                self.var.CurrentCropArea),
            axis=(0,1))         # FIXME
        self.var.th = np.copy(thnew)

# class Drainage(object):
#     """Class to infiltrate incoming water"""
    
#     def __init__(self, Drainage_variable):
#         self.var = Drainage_variable

#     def initial(self):
#         self.var.FluxOut = np.zeros((self.var.nFarm, self.var.nCrop, self.var.nComp, self.var.nCell))
#         self.var.DeepPerc = np.zeros((self.var.nFarm, self.var.nCrop, self.var.nCell))
#         self.var.Recharge = np.zeros((self.var.nCell))

#         # experiment
#         # self.var.th_exc_comp = self.var.th_sat_comp - self.var.th_fc_comp
#         # self.var.exp_th_exc_comp = np.exp(self.var.th_exc_comp)
        
#     # TODO: install numba and try this
#     @staticmethod
#     @jit(nopython=True, cache=True)
#     def compute_dthdt(th, th_sat, th_fc, th_fc_adj, tau):
#         cond1 = th <= th_fc_adj
#         dthdt1 = np.zeros_like(th, dtype=np.float64)
#         dthdt = (dthdt1 * cond1)
#         cond2 = np.logical_not(cond1) & (th >= th_sat)
#         dthdt2 = tau * (th_sat - th_fc)
#         dthdt += (dthdt2 * cond2)
#         cond3 = np.logical_not(cond1 | cond2)
#         dthdt3 = (tau * (th_sat - th_fc) * ((np.exp(th - th_fc) - 1) / (np.exp(th_sat - th_fc) - 1)))
#         dthdt += (dthdt3 * cond3)
#         cond4 = (cond2 | cond3) & ((th - dthdt) < th_fc_adj)
#         dthdt4 = (th - th_fc_adj)
#         dthdt += (dthdt4 * cond4)
#         return dthdt
        
#     def dynamic(self):
#         """Function to redistribute stored soil water"""

#         # print(self.var.th[...,0,1])
#         # print(self.var.DeepPerc[...,1])
#         thh = np.asfortranarray(self.var.th)
#         layer_ix = self.var.layerIndex + 1
#         aquacrop_fc.drainage_w.update_drainage_w(
#             thh,
#             # np.asfortranarray(self.var.th),
#             np.asfortranarray(self.var.DeepPerc),
#             np.asfortranarray(self.var.FluxOut),
#             np.asfortranarray(self.var.th_sat),
#             np.asfortranarray(self.var.th_fc),
#             np.asfortranarray(self.var.k_sat),
#             np.asfortranarray(self.var.tau),
#             np.asfortranarray(self.var.th_fc_adj),
#             np.asfortranarray(self.var.dz),
#             np.asfortranarray(self.var.dz_sum),
#             np.asfortranarray(layer_ix),
#             self.var.nFarm, self.var.nCrop, self.var.nComp, self.var.nLayer, self.var.nCell
#             )
#         self.var.th = np.ascontiguousarray(thh).copy()
#         # print(np.ascontiguousarray(thh)[...,0,1])
#         # print(self.var.th[...,0,1])
#         # print(self.var.DeepPerc[...,1])
        
#         # self.var.Recharge = np.sum(
#         #     np.multiply(
#         #         (self.var.DeepPerc / 1000.),
#         #         self.var.CurrentCropArea),
#         #     axis=(0,1))         # FIXME
            
#         # # dims = self.var.th.shape
#         # thnew = np.copy(self.var.th)
#         # drainsum = np.zeros((self.var.nFarm, self.var.nCrop, self.var.nCell))
#         # for comp in range(self.var.nComp):

#         #     # Calculate drainage ability of compartment ii
#         #     dthdt = self.compute_dthdt(
#         #         self.var.th[:,:,comp,:],
#         #         self.var.th_sat_comp[:,:,comp,:],
#         #         self.var.th_fc_comp[:,:,comp,:],
#         #         self.var.th_fc_adj[:,:,comp,:],
#         #         self.var.tau_comp[:,:,comp,:]
#         #     )

#         #     # Drainage from compartment ii (mm) (Line 41 in AOS_Drainage.m)
#         #     draincomp = dthdt * self.var.dz[comp] * 1000

#         #     # Check drainage ability of compartment ii against cumulative
#         #     # drainage from compartments above (Lines 45-52 in AOS_Drainage.m)
#         #     excess = np.zeros((self.var.nFarm, self.var.nCrop, self.var.nCell))
#         #     prethick = self.var.dz_sum[comp] - self.var.dz[comp]
#         #     drainmax = dthdt * 1000 * prethick
#         #     drainability = (drainsum <= drainmax)

#         #     # Drain compartment
#         #     cond5 = drainability
#         #     thnew[:,:,comp,:][cond5] = (self.var.th[:,:,comp,:] - dthdt)[cond5]

#         #     # Update cumulative drainage (mm), restrict to saturated hydraulic
#         #     # conductivity and adjust excess drainage flow
#         #     drainsum[cond5] += draincomp[cond5]
#         #     cond51 = (cond5 & (drainsum > self.var.k_sat_comp[:,:,comp,:]))
#         #     excess[cond51] += (drainsum - self.var.k_sat_comp[:,:,comp,:])[cond51]
#         #     drainsum[cond51] = self.var.k_sat_comp[:,:,comp,:][cond51]

#         #     # Calculate value of theta (thX) needed to provide a drainage
#         #     # ability equal to cumulative drainage (Lines 70-85 in AOS_Drainage.m)

#         #     # TODO: put this in a staticmethod, use numba            
#         #     cond6 = np.logical_not(drainability)
#         #     dthdt[cond6] = np.divide(drainsum, 1000 * prethick, out=np.zeros_like(drainsum), where=prethick!=0)[cond6]
#         #     thX = np.zeros((self.var.nFarm, self.var.nCrop, self.var.nCell))
#         #     cond61 = (cond6 & (dthdt <= 0))
#         #     thX[cond61] = self.var.th_fc_adj[:,:,comp,:][cond61]
#         #     cond62 = (cond6 & np.logical_not(cond61) & (self.var.tau_comp[:,:,comp,:] > 0))
#         #     A = (1 + ((dthdt * (np.exp(self.var.th_sat_comp[:,:,comp,:] - self.var.th_fc_comp[:,:,comp,:]) - 1)) / (self.var.tau_comp[:,:,comp,:] * (self.var.th_sat_comp[:,:,comp,:] - self.var.th_fc_comp[:,:,comp,:]))))
#         #     thX[cond62] = (self.var.th_fc_adj[:,:,comp,:] + np.log(A))[cond62]
#         #     thX[cond62] = np.clip(thX, self.var.th_fc_adj[:,:,comp,:], None)[cond62]
#         #     cond63 = (cond6 & np.logical_not(cond61 | cond62))
#         #     thX[cond63] = (self.var.th_sat_comp[:,:,comp,:] + 0.01)[cond63]

#         #     # Check thX against hydraulic properties of current soil layer

#         #     # Increase compartment ii water content with cumulative drainage
#         #     cond64 = (cond6 & (thX <= self.var.th_sat_comp[:,:,comp,:]))
#         #     thnew[:,:,comp,:][cond64] = (self.var.th[:,:,comp,:] + (drainsum / (1000 * self.var.dz[comp])))[cond64]

#         #     # Cumulative drainage is the drainage difference between theta_x and
#         #     # new theta plus drainage ability at theta_x
#         #     cond641 = (cond64 & (thnew[:,:,comp,:] > thX))
#         #     drainsum[cond641] = ((thnew[:,:,comp,:] - thX) * 1000 * self.var.dz[comp])[cond641]

#         #     # Calculate drainage ability for thX
#         #     dthdt = self.compute_dthdt(
#         #         thX,
#         #         self.var.th_sat_comp[:,:,comp,:],
#         #         self.var.th_fc_comp[:,:,comp,:],
#         #         self.var.th_fc_adj[:,:,comp,:],
#         #         self.var.tau_comp[:,:,comp,:]
#         #     )

#         #     # Update cumulative drainage (mm), restrict to saturated hydraulic
#         #     # conductivity and adjust excess drainage flow
#         #     drainsum[cond641] += (dthdt * 1000 * self.var.dz[comp])[cond641]
#         #     cond6415 = (cond641 & (drainsum > self.var.k_sat_comp[:,:,comp,:]))
#         #     excess[cond6415] += (drainsum - self.var.k_sat_comp[:,:,comp,:])[cond6415]
#         #     drainsum[cond6415] = self.var.k_sat_comp[:,:,comp,:][cond6415]

#         #     # Update water content
#         #     thnew[:,:,comp,:][cond641] = (thX - dthdt)[cond641]

#         #     # Calculate drainage ability for updated water content
#         #     cond642 = (cond64 & np.logical_not(cond641) & (thnew[:,:,comp,:] > self.var.th_fc_adj[:,:,comp,:]))
#         #     dthdt = self.compute_dthdt(
#         #         thnew[:,:,comp,:],
#         #         self.var.th_sat_comp[:,:,comp,:],
#         #         self.var.th_fc_comp[:,:,comp,:],
#         #         self.var.th_fc_adj[:,:,comp,:],
#         #         self.var.tau_comp[:,:,comp,:]
#         #     )

#         #     # Update water content
#         #     thnew[:,:,comp,:][cond642] = (thnew[:,:,comp,:] - dthdt)[cond642]

#         #     # Update cumulative drainage (mm), restrict to saturated hydraulic
#         #     # conductivity and adjust excess drainage flow
#         #     drainsum[cond642] = (dthdt * 1000 * self.var.dz[comp])[cond642]
#         #     cond6425 = (cond642 & (drainsum > self.var.k_sat_comp[:,:,comp,:]))
#         #     excess[cond6425] += (drainsum - self.var.k_sat_comp[:,:,comp,:])[cond6425]
#         #     drainsum[cond6425] = self.var.k_sat_comp[:,:,comp,:][cond6425]

#         #     # Otherwise, drainage is zero
#         #     cond643 = (cond64 & np.logical_not(cond641 | cond642))
#         #     drainsum[cond643] = 0

#         #     # Increase water content in compartment ii with cumulative
#         #     # drainage from above
#         #     cond65 = (cond6 & np.logical_not(cond64) & (thX > self.var.th_sat_comp[:,:,comp,:]))
#         #     thnew[:,:,comp,:][cond65] = (self.var.th[:,:,comp,:] + (drainsum / (1000 * self.var.dz[comp])))[cond65]

#         #     # Check new water content against hydraulic properties of soil layer
#         #     # Lines 166-198
#         #     cond651 = (cond65 & (thnew[:,:,comp,:] <= self.var.th_sat_comp[:,:,comp,:]))

#         #     # Calculate new drainage ability
#         #     cond6511 = (cond651 & (thnew[:,:,comp,:] > self.var.th_fc_adj[:,:,comp,:]))
#         #     dthdt = self.compute_dthdt(
#         #         thnew[:,:,comp,:],
#         #         self.var.th_sat_comp[:,:,comp,:],
#         #         self.var.th_fc_comp[:,:,comp,:],
#         #         self.var.th_fc_adj[:,:,comp,:],
#         #         self.var.tau_comp[:,:,comp,:]
#         #     )

#         #     # Update water content
#         #     thnew[:,:,comp,:][cond6511] -= (dthdt)[cond6511]

#         #     # Update cumulative drainage (mm), restrict to saturated hydraulic
#         #     # conductivity and adjust excess drainage flow
#         #     drainsum[cond6511] = (dthdt * 1000 * self.var.dz[comp])[cond6511]
#         #     cond65115 = (cond6511 & (drainsum > self.var.k_sat_comp[:,:,comp,:]))
#         #     excess[cond65115] += (drainsum - self.var.k_sat_comp[:,:,comp,:])[cond65115]
#         #     drainsum[cond65115] = self.var.k_sat_comp[:,:,comp,:][cond65115]

#         #     cond6512 = (cond651 & (np.logical_not(cond6511)))
#         #     drainsum[cond6512] = 0

#         #     # Calculate excess drainage above saturation
#         #     cond652 = (cond65 & np.logical_not(cond651) & (thnew[:,:,comp,:] > self.var.th_sat_comp[:,:,comp,:]))
#         #     excess[cond652] = ((thnew[:,:,comp,:] - self.var.th_sat_comp[:,:,comp,:]) * 1000 * self.var.dz[comp])[cond652]

#         #     # Calculate drainage ability for updated water content
#         #     dthdt = self.compute_dthdt(
#         #         thnew[:,:,comp,:],
#         #         self.var.th_sat_comp[:,:,comp,:],
#         #         self.var.th_fc_comp[:,:,comp,:],
#         #         self.var.th_fc_adj[:,:,comp,:],
#         #         self.var.tau_comp[:,:,comp,:]
#         #     )

#         #     # Update water content
#         #     thnew[:,:,comp,:][cond652] = (self.var.th_sat_comp[:,:,comp,:] - dthdt)[cond652]

#         #     # Update drainage, maximum drainage, excess drainage
#         #     draincomp[cond652] = (dthdt * 1000 * self.var.dz[comp])[cond652]
#         #     drainmax[cond652] = (dthdt * 1000 * prethick)[cond652]
#         #     drainmax[cond652] = np.clip(drainmax, None, excess)[cond652]
#         #     excess[cond652] -= drainmax[cond652]

#         #     # Update cumulative drainage (mm), restrict to saturated hydraulic
#         #     # conductivity and adjust excess drainage flow
#         #     drainsum[cond652] = (draincomp + drainmax)[cond652]
#         #     cond6525 = (cond652 & (drainsum > self.var.k_sat_comp[:,:,comp,:]))
#         #     excess[cond6525] += (drainsum - self.var.k_sat_comp[:,:,comp,:])[cond6525]
#         #     drainsum[cond6525] = self.var.k_sat_comp[:,:,comp,:][cond6525]

#         #     # Store output flux from compartment ii
#         #     self.var.FluxOut[:,:,comp,:] = np.copy(drainsum)

#         #     # TODO: make a staticmethod for this loop?
            
#         #     # Redistribute excess in compartment above
#         #     precomp = comp + 1
#         #     while (np.any(excess > 0)) & (precomp != 0):

#         #         # Include condition here so that it is updated
#         #         cond7 = (excess > 0)

#         #         # Update compartment counter
#         #         precomp -= 1

#         #         # Update flux from compartment
#         #         if (precomp < comp):
#         #             self.var.FluxOut[:,:,precomp,:][cond7] -= excess[cond7]

#         #         # Increase water content to store excess
#         #         thnew[:,:,precomp,:][cond7] += (excess / (1000 * self.var.dz[precomp]))[cond7]

#         #         # Limit water content to saturation and adjust excess counter
#         #         cond71 = (cond7 & (thnew[:,:,precomp,:] > self.var.th_sat_comp[:,:,precomp,:]))
#         #         excess[cond71] = ((thnew[:,:,precomp,:] - self.var.th_sat_comp[:,:,precomp,:]) * 1000 * self.var.dz[precomp])[cond71]
#         #         thnew[:,:,precomp,:][cond71] = self.var.th_sat_comp[:,:,precomp,:][cond71]

#         #         cond72 = (cond7 & np.logical_not(cond71))
#         #         excess[cond72] = 0

#         # self.var.DeepPerc = np.copy(drainsum)
#         # self.var.Recharge = np.sum(
#         #     np.multiply(
#         #         (drainsum / 1000.),
#         #         self.var.CurrentCropArea),
#         #     axis=(0,1))         # FIXME
#         # self.var.th = np.copy(thnew)
        
#         # print(self.var.th[...,0,1])
#         # print(self.var.DeepPerc[...,1])
