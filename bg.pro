function colour_correction, ccmaps, tdvec, betavec, t_bg, beta, ind

  ii = interpol(indgen(n_elements(tdvec)), tdvec, T_bg)
  jj = interpol(indgen(n_elements(betavec)), betavec, beta)
  cc = interpolate(ccmaps, ii, jj, ind, /grid)

  return, cc

end

;==================================================================
function bg, x, a, tdvec=tdvec, betavec=betavec, ccmaps=ccmaps, reflambda=x0, opacity=opacity
  
; computes modified black body dust spectrum 
;
; x : fltarr; wavelength (micron)
; a : model parameter. 3 values = compute a single beta model, 
;     5 values = two beta model
;     6 values = two modified black bodies
;
; MAMD
; 22/08/2010 : creation
; 15/02/2013 : add colour correction and 2 modified black body model
; 05/03/2015 : add opacity


if not keyword_set(x0) then x0=850.  ; micron

E_bg = a[0]
T_bg1 = a[1]
beta1 = a[2]
nbparam = n_elements(a)
if (nbparam gt 3) then begin
   beta2 = a[nbparam-1]
   if (nbparam eq 5) then begin
      T_bg2 = T_bg1^((4+beta1)/(4+beta2))
   endif else begin
      T_bg2 = a[4]
   endelse
endif

; compute colour correction
cc1 = fltarr(n_elements(x))
cc2 = fltarr(n_elements(x))
if keyword_set(tdvec) then begin ; colour correction
   indx = indgen(n_elements(x))
   cc1 = colour_correction(ccmaps, tdvec, betavec, t_bg1, beta1, indx)
   if (nbparam gt 3) then cc2 = colour_correction(ccmaps, tdvec, betavec, t_bg2, beta2, indx)
endif else begin
   cc1[*] = 1.
   cc2[*] = 1.
endelse

; compute model
case nbparam of
   3: begin   ; one MBB
      model = 1.e20*E_bg*(x/x0)^(-1*Beta1)*bnu_planck(x, T_bg1)
      if keyword_set(opacity) then begin
         tau_nu = E_bg*(x/x0)^(-1*Beta1)
         model = model*exp(-tau_nu)
      endif
      model = model*reform(cc1) ; colour correction
   end
   5: begin   ; two MBB sharing the same temperature
      model = 1.e20*E_bg*(x/x0)^(-1*Beta1)*bnu_planck(x, T_bg1)      
      if keyword_set(opacity) then begin
         tau_nu = E_bg*(x/x0)^(-1*Beta1)
         model = model*exp(-tau_nu)
      endif
      model = model*reform(cc1)
      E_bg = a[3]
      model2 = 1.e20*E_bg*(x/x0)^(-1*Beta2)*bnu_planck(x, T_bg2)
      if keyword_set(opacity) then begin
         tau_nu = E_bg*(x/x0)^(-1*Beta2)
         model2 = model2*exp(-tau_nu)
      endif
      model2 = model2*reform(cc2)
      model = model + model2
   end
   6: begin   ; two MBB
      model = 1.e20*E_bg*(x/x0)^(-1*Beta1)*bnu_planck(x, T_bg1)      
      if keyword_set(opacity) then begin
         tau_nu = E_bg*(x/x0)^(-1*Beta1)
         model = model*exp(-tau_nu)
      endif
      model = model*reform(cc1)
      E_bg = a[3]
      model2 = 1.e20*E_bg*(x/x0)^(-1*Beta2)*bnu_planck(x, T_bg2)
      if keyword_set(opacity) then begin
         tau_nu = E_bg*(x/x0)^(-1*Beta2)
         model2 = model2*exp(-tau_nu)
      endif
      model2 = model2*reform(cc2)
      model = model + model2
   end
endcase

return, model

end

