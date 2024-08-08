# ***********************************************************************
# ******************  CANADIAN ASTRONOMY DATA CENTRE  *******************
# *************  CENTRE CANADIEN DE DONNÉES ASTRONOMIQUES  **************
#
#  (c) 2024.                            (c) 2024.
#  Government of Canada                 Gouvernement du Canada
#  National Research Council            Conseil national de recherches
#  Ottawa, Canada, K1A 0R6              Ottawa, Canada, K1A 0R6
#  All rights reserved                  Tous droits réservés
#
#  NRC disclaims any warranties,        Le CNRC dénie toute garantie
#  expressed, implied, or               énoncée, implicite ou légale,
#  statutory, of any kind with          de quelque nature que ce
#  respect to the software,             soit, concernant le logiciel,
#  including without limitation         y compris sans restriction
#  any warranty of merchantability      toute garantie de valeur
#  or fitness for a particular          marchande ou de pertinence
#  purpose. NRC shall not be            pour un usage particulier.
#  liable in any event for any          Le CNRC ne pourra en aucun cas
#  damages, whether direct or           être tenu responsable de tout
#  indirect, special or general,        dommage, direct ou indirect,
#  consequential or incidental,         particulier ou général,
#  arising from the use of the          accessoire ou fortuit, résultant
#  software.  Neither the name          de l'utilisation du logiciel. Ni
#  of the National Research             le nom du Conseil National de
#  Council of Canada nor the            Recherches du Canada ni les noms
#  names of its contributors may        de ses  participants ne peuvent
#  be used to endorse or promote        être utilisés pour approuver ou
#  products derived from this           promouvoir les produits dérivés
#  software without specific prior      de ce logiciel sans autorisation
#  written permission.                  préalable et particulière
#                                       par écrit.
#
#  This file is part of the             Ce fichier fait partie du projet
#  OpenCADC project.                    OpenCADC.
#
#  OpenCADC is free software:           OpenCADC est un logiciel libre ;
#  you can redistribute it and/or       vous pouvez le redistribuer ou le
#  modify it under the terms of         modifier suivant les termes de
#  the GNU Affero General Public        la “GNU Affero General Public
#  License as published by the          License” telle que publiée
#  Free Software Foundation,            par la Free Software Foundation
#  either version 3 of the              : soit la version 3 de cette
#  License, or (at your option)         licence, soit (à votre gré)
#  any later version.                   toute version ultérieure.
#
#  OpenCADC is distributed in the       OpenCADC est distribué
#  hope that it will be useful,         dans l’espoir qu’il vous
#  but WITHOUT ANY WARRANTY;            sera utile, mais SANS AUCUNE
#  without even the implied             GARANTIE : sans même la garantie
#  warranty of MERCHANTABILITY          implicite de COMMERCIALISABILITÉ
#  or FITNESS FOR A PARTICULAR          ni d’ADÉQUATION À UN OBJECTIF
#  PURPOSE.  See the GNU Affero         PARTICULIER. Consultez la Licence
#  General Public License for           Générale Publique GNU Affero
#  more details.                        pour plus de détails.
#
#  You should have received             Vous devriez avoir reçu une
#  a copy of the GNU Affero             copie de la Licence Générale
#  General Public License along         Publique GNU Affero avec
#  with OpenCADC.  If not, see          OpenCADC ; si ce n’est
#  <http://www.gnu.org/licenses/>.      pas le cas, consultez :
#                                       <http://www.gnu.org/licenses/>.
#
#  $Revision: 4 $
#
# ***********************************************************************
#

from os.path import basename
from urllib import parse as parse
from caom2pipe.manage_composable import CadcException, StorageName


class PossumName(StorageName):
    """
    From AusSRC:
    PSM_pilot1_944MHz_18asec_2226-5552_11268_i_v1.fits
    PSM_pilot1_944MHz_18asec_2226-5552_11268_q_v1.fits
    PSM_pilot1_944MHz_18asec_2226-5552_11268_u_v1.fits

    From POSSUM Group:
    PSM_pilot1_944MHz_18asec_2226-5552_11268_p3d_v1_ampPeakPIfitEff.fits
    PSM_pilot1_944MHz_18asec_2226-5552_11268_p3d_v1_coeff0err.fits
    PSM_pilot1_944MHz_18asec_2226-5552_11268_p3d_v1_coeff0.fits
    PSM_pilot1_944MHz_18asec_2226-5552_11268_p3d_v1_coeff1err.fits
    PSM_pilot1_944MHz_18asec_2226-5552_11268_p3d_v1_coeff1.fits
    PSM_pilot1_944MHz_18asec_2226-5552_11268_p3d_v1_dAmpPeakPIfit.fits
    PSM_pilot1_944MHz_18asec_2226-5552_11268_p3d_v1_dFDFcorMAD.fits
    PSM_pilot1_944MHz_18asec_2226-5552_11268_p3d_v1_dPhiPeakPIfit_rm2.fits
    PSM_pilot1_944MHz_18asec_2226-5552_11268_p3d_v1_dPolAngle0Fit_deg.fits
    PSM_pilot1_944MHz_18asec_2226-5552_11268_p3d_v1_FDF_imag_dirty.fits
    PSM_pilot1_944MHz_18asec_2226-5552_11268_p3d_v1_FDF_real_dirty.fits
    PSM_pilot1_944MHz_18asec_2226-5552_11268_p3d_v1_FDF_tot_dirty.fits
    PSM_pilot1_944MHz_18asec_2226-5552_11268_p3d_v1_lam0Sq_m2.fits
    PSM_pilot1_944MHz_18asec_2226-5552_11268_p3d_v1_mad_chanwidth.fits
    PSM_pilot1_944MHz_18asec_2226-5552_11268_p3d_v1_max_freq.fits
    PSM_pilot1_944MHz_18asec_2226-5552_11268_p3d_v1_min_freq.fits
    PSM_pilot1_944MHz_18asec_2226-5552_11268_p3d_v1_peakFDFimagFit.fits
    PSM_pilot1_944MHz_18asec_2226-5552_11268_p3d_v1_peakFDFrealFit.fits
    PSM_pilot1_944MHz_18asec_2226-5552_11268_p3d_v1_phiPeakPIfit_rm2.fits
    PSM_pilot1_944MHz_18asec_2226-5552_11268_p3d_v1_polAngle0Fit_deg.fits
    PSM_pilot1_944MHz_18asec_2226-5552_11268_p3d_v1_RMSF1D.fits
    PSM_pilot1_944MHz_18asec_2226-5552_11268_p3d_v1_RMSF_FWHM.fits
    PSM_pilot1_944MHz_18asec_2226-5552_11268_p3d_v1_snrPIfit.fits

    Guessing on the rename for this:
    PSM_pilot1_944MHz_18asec_2226-5552_11268_p3d_v1_RMSF_imag.fits.tar.gz
    PSM_pilot1_944MHz_18asec_2226-5552_11268_p3d_v1_RMSF_real.fits.tar.gz
    to:
    PSM_pilot1_944MHz_18asec_2226-5552_11268_p3d_v1_RMSF_imag.fits.fz
    PSM_pilot1_944MHz_18asec_2226-5552_11268_p3d_v1_RMSF_real.fits.fz

    """

    POSSUM_NAME_PATTERN = '*'

    def __init__(self, entry):
        self._central_frequency = None
        self._healpix_index = None
        self._pilot = None
        self._position = None
        self._resolution = None
        self._version = None
        super().__init__(file_name=basename(entry.replace('.header', '')), source_names=[entry])

    def __str__(self):
        temp = super().__str__()
        return f'{temp}\n          rename: {self.rename("")}\n   healpix_index: {self._healpix_index}\n      resolution: {self._resolution}'

    @property
    def file_uri(self):
        """The CADC Storage URI for the file."""
        current_scheme = self._get_scheme()
        return self._get_uri(self._file_name.replace('.gz', '').replace('.header', ''), current_scheme)

    @property
    def healpix_index(self):
        return self._healpix_index

    @property
    def is_bintable(self):
        return '_FDFs.fits' in self._file_name or '_spectra.fits' in self._file_name

    @property
    def is_original(self):
        return 'asec' not in self._file_id

    @property
    def prev(self):
        """The preview file name for the file."""
        return f'{self._obs_id}_{self._product_id}_prev.jpg'

    @property
    def spatial_resolution(self):
        return float(self._resolution.replace('asec', ''))

    @property
    def thumb(self):
        """The thumbnail file name for the file."""
        return f'{self._obs_id}_{self._product_id}_prev_256.jpg'

    def rename(self, band, version='v1', resolution='20asec'):
        if 'asec' in self._file_id:
            self._logger.info(f'{self._file_id} has already been renamed.')
            return self._file_name
        # Jennifer West, 08-04-24
        # band1 == 944MHz, band2 == 1296MHz
        # is version == v1
        # resolution == 20asec
        # Jennifer West 16-04-24
        # make the coordinates 8 digits => XXXX-XXXX
        x = self._file_name.split('.fits')
        bits = x[0].split('.')
        for index, bit in enumerate(bits):
            if '-' in bit or '+' in bit:
                teeny_bits = bit.split('_')
                for teeny_index, teeny_bit in enumerate(teeny_bits):
                    marker = None
                    if '-' in teeny_bit:
                        teenier_bits = teeny_bit.split('-')
                        marker = '-'
                    elif '+' in teeny_bit:
                        teenier_bits = teeny_bit.split('+')
                        marker = '+'
                    for ii in [0, 1]:
                        if len(teenier_bits[ii]) != 4:
                            teenier_bits[ii] = f'{teenier_bits[ii]:04}'
                    teeny_bits[teeny_index] = f'{teenier_bits[0]}{marker}{teenier_bits[1]}'
                bits[index] = '_'.join(jj for jj in teeny_bits)
        temp1 = '_'.join(ii for ii in bits)
        if 'band1' in self._file_name:
            y = temp1.replace('band1', f'944MHz_{resolution}')
        elif 'band2' in self._file_name:
            y = temp1.replace('band2', f'1296MHz_{resolution}')
        else:
            y = temp1
        temp2 = y.replace('POSSUM', 'PSM').replace('mfs_', '')
        if len(x) == 1:
            temp = f'{temp2}_{version}.fits'
        else:
            temp = f'{temp2}_{version}.fits{x[1]}'
        return temp

    def set_destination_uris(self):
        for entry in self._source_names:
            temp = parse.urlparse(entry)
            base_name = basename(temp.path)
            current_scheme = StorageName.scheme
            if '_p3d_' in base_name or '_p1d_' in base_name:
                current_scheme = StorageName.preview_scheme
            if '.fits' in entry:
                self._destination_uris.append(
                    self._get_uri(base_name.replace('.gz', '').replace('.header', ''), current_scheme)
                )
            else:
                self._destination_uris.append(self._get_uri(base_name, current_scheme))

    def set_obs_id(self):
        # leave off the "PSM" because collection is POSSUM
        if self.is_original:
            # how the file names come from Pawsey
            self._obs_id = self._file_id
        else:
            bits = self._file_id.split('_')
            index = lambda x: bits[x + 1] if 'pilot' in self._file_id else bits[x]
            self._central_frequency = index(1)
            self._resolution = index(2)
            self._position = index(3)
            self._healpix_index = index(4)
            for bit in bits:
                if bit.startswith('v') and len(bit) in [2, 3]:
                    self._version = bit
                if bit.startswith('pilot'):
                    self._pilot = bit
            if self._pilot:
                self._obs_id = (
                    f'{self._central_frequency}_{self._resolution}_{self._position}_{self._healpix_index}_'
                    f'{self._pilot}_{self._version}'
                )
            else:
                self._obs_id = (
                    f'{self._central_frequency}_{self._resolution}_{self._position}_{self._healpix_index}_{self._version}'
                )

    def set_product_id(self):
        if self.is_original:
            self._product_id = self._file_id
        else:
            bits = self._file_id.split('_')
            if '_p3d_' in self._file_id:
                if 'FDF_im_dirty' in self._file_id:
                    self._product_id = 'FDF_im_dirty_3d_pipeline'
                elif 'FDF_real_dirty' in self._file_id:
                    self._product_id = 'FDF_real_dirty_3d_pipeline'
                elif 'FDF_tot_dirty' in self._file_id:
                    self._product_id = 'FDF_tot_dirty_3d_pipeline'
                elif 'FDF_tot_dirty' in self._file_id:
                    self._product_id = 'RMSF_FWHM_3d_pipeline'
                else:
                    self._product_id = '3d_pipeline'
            elif '_p1d_' in self._file_id:
                # catalog in csv, spectra, FDF in BINTABLE
                self._product_id = '1d_pipeline'
            elif 't0' in bits or 't1' in bits:
                # Cameron Van Eck - 23-10-23
                # “mfs_i_t0" or “multifrequencysynthesis_i_t0” for the image product ProductID
                self._product_id = f'multifrequencysynthesis_{bits[-2]}_{bits[-3]}'
            elif 'i' in bits:
                self._product_id = 'raw_i'
            elif 'q' in bits or 'u' in bits:
                self._product_id = 'raw_qu'
        if self._product_id is None:
            raise CadcException(f'Unexpected raw file naming pattern {self._file_id}')

    def un_name(self):
        """Undo the renaming from Pawsey Acacia, to be able to work backwards for sc2 Observation metadata."""
        return self._obs_id.split('asec_')[1].split('_v1')[0]

    def _get_scheme(self):
        if self._product_id in ['raw_i', 'raw_qu'] or self._product_id.startswith('multifrequencysynthesis_'):
            result = StorageName.scheme
        else:
            result = StorageName.preview_scheme
        return result
