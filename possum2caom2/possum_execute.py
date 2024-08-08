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

import glob
import logging
import os

import numpy as np
from astropy import units
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from astropy_healpix import HEALPix

from caom2pipe.client_composable import ClientCollection
from caom2pipe.data_source_composable import RemoteRcloneIncrementalDataSource
from caom2pipe.manage_composable import Config, exec_cmd
from caom2pipe.manage_composable import ExecutionReporter, Observable, StorageName, TaskType
from caom2pipe.name_builder_composable import EntryBuilder
from caom2pipe.reader_composable import FileMetadataReader, RemoteRcloneMetadataReader
from caom2pipe.remote_composable import ExecutionUnit, ExecutionUnitOrganizeExecutes
from caom2pipe.run_composable import ExecutionUnitStateRunner, set_logging
from caom2repo import CAOM2RepoClient
from possum2caom2 import fits2caom2_augmentation, preview_augmentation, spectral_augmentation
from possum2caom2.storage_name import PossumName


__all__ = ['DATA_VISITORS', 'META_VISITORS', 'remote_execution']
META_VISITORS = [fits2caom2_augmentation]
DATA_VISITORS = [preview_augmentation, spectral_augmentation]


class RCloneClients(ClientCollection):

    def __init__(self, config):
        super().__init__(config)
        # TODO rclone credentials
        self._rclone_client = None
        if TaskType.SCRAPE in config.task_types:
            self._logger.info(f'SCRAPE\'ing data - no clients will be initialized.')
        else:
            self._server_side_ctor_client = CAOM2RepoClient(
                self._subject, config.logging_level, config.server_side_resource_id
            )

    @property
    def rclone_client(self):
        return self._rclone_client

    @property
    def server_side_ctor_client(self):
        return self._server_side_ctor_client


class RemoteIncrementalDataSource(RemoteRcloneIncrementalDataSource):

    def __init__(self, config, start_key, metadata_reader, **kwargs):
        super().__init__(config, start_key, metadata_reader, **kwargs)

    def get_time_box_work(self, prev_exec_dt, exec_dt):
        self._logger.debug('Begin get_time_box_work')
        self._kwargs['prev_exec_dt'] = prev_exec_dt
        self._kwargs['exec_dt'] = exec_dt
        self._kwargs['metadata_reader'] = self._metadata_reader
        self._kwargs['builder'] = EntryBuilder(PossumName)
        self._kwargs['meta_visitors'] = META_VISITORS
        self._kwargs['data_visitors'] = DATA_VISITORS
        self._kwargs['staged_metadata_reader'] = FileMetadataReader()
        execution_unit = PossumExecutionUnit(self._config, **self._kwargs)
        execution_unit.num_entries, execution_unit.entry_dt = self._metadata_reader.get_time_box_work_parameters(prev_exec_dt, exec_dt)
        if execution_unit.num_entries > 0:
            execution_unit.start()
            # get the files from the DataSource to the staging space
            # --max-age  -> only transfer files younger than this in s
            # --min-age -> only transfer files older than this in s
            exec_cmd(
                f'rclone copy {self._remote_key} {execution_unit.working_directory} --max-age={prev_exec_dt.isoformat()} '
                f'--min-age={exec_dt.isoformat()} --include={self._data_source_extensions}'
            )
        self._logger.debug('End get_time_box_work')
        return execution_unit


class PossumExecutionUnit(ExecutionUnit):

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

    def _RADEC_hms_dms_to_string(self, c: SkyCoord):
        """
        Round SkyCoordinate "c" (RA,DEC) to a string of "{hhmm}{ddmm}" for "{ra}{dec}"
        with proper rounding
        """
        ### round RA ###
        ra_h, ra_m, ra_s = c.ra.hms
        # round minutes based on seconds
        ra_m = round(ra_m + (ra_s / 60.0))
        # round hours based on minutes
        if ra_m >= 60:
            ra_m = 0
            ra_h += 1
        # check hours boundary
        if ra_h >= 24:
            ra_h = 0
        # create RA hh mm string with zero padding
        ra_hh_mm = f"{int(ra_h):02d}{int(ra_m):02d}"

        ### round DEC ###
        dsign = "+" # rounding absolute dec and fixing sign after
        if c.dec < (0*units.degree):
            dsign = "-"
        dec_d, dec_m, dec_s = np.abs(c.dec.dms)
        # round minutes based on seconds
        dec_m = round(abs(dec_m) + abs(dec_s)/60.0)
        # round degrees based on minutes
        if dec_m >= 60:
            dec_m = 0
            dec_d += 1
        # create DEC dd mm string with zero padding and the proper sign
        dec_dd_mm = f"{dsign}{int(dec_d):02d}{int(dec_m):02d}"

        # Create final string "hhmm-ddmm"
        RADEC = f"{ra_hh_mm}{dec_dd_mm}"
        return RADEC

    def _find_new_file_name(self, hdr, mfs):
    # def name(fitsimage, prefix, version="v1", mfs=False):
        """
        Algorithm from @Sebokolodi, @Cameron-Van-Eck, @ErikOsinga (github).
        Setting up the name to be used for tiles. The script reads the bmaj and stokes from the fits header. The
        rest of the parameters are flexible to change.

        fitsimage: tile image
        prefix   : prefix to use. E.g. PSM for full survey,
                PSM_pilot1 for POSSUM pilot 1
                PSM_pilot2 for POSSUM pilot 2
        tileID   : tile pixel (Healpix pixel)

        version  : version of the output product. Version 1 is v1, version is v2,
                and so forth.

        """

        self._logger.debug('Begin _find_new_file_name')

        # get bmaj.
        bmaj = round(hdr.get('BMAJ') * 3600.0)
        if bmaj:
            bmaj =  f'{bmaj:2d}asec'

        # extract stokes parameter. It can be in either the 3rd or fourth axis.

        if hdr.get('CTYPE3') == 'STOKES':
            stokes = hdr.get('CRVAL3')
            # if Stokes is axis 3, then frequency is axis 4.
            freq0 = hdr.get('CRVAL4')
            dfreq = hdr.get('CDELT4')
            n = hdr.get('NAXIS4')
            if n and n > 1:
                cenfreq = round((freq0 + (freq0 + (n - 1) * dfreq))/(2.0 * 1e6))
            else:
                cenfreq = round(freq0/1e6)


        elif hdr.get('CTYPE4') == 'STOKES':
            stokes = hdr.get('CRVAL4')
            # if Stokes is axis 4, then frequency is axis 3. If we have >4 axis, the script will fail.
            freq0 = hdr.get('CRVAL3')
            dfreq = hdr.get('CDELT3')
            n = hdr.get('NAXIS3')
            if n and n > 1:
                cenfreq = round((freq0 + (freq0 + (n - 1) * dfreq))/(2.0 * 1e6))
            else:
                cenfreq = round(freq0/1e6)

        else:
            self._logger.error('Cannot find Stokes axis on the 3rd/4th axis')
            return None

        cenfreq = f'{round(cenfreq)}MHz'

        # stokes I=1, Q=2, U=3 and 4=V
        if int(stokes) == 1:
            stokesid = 'i'

        elif int(stokes) == 2:
            stokesid = 'q'

        elif int(stokes) == 3:
            stokesid = 'u'

        elif int(stokes) == 4:
            stokesid = 'v'

        self._logger.info('Define healpix grid for nside 32')
        # define the healpix grid
        hp = HEALPix(nside=32, order='ring', frame='icrs')

        # read the image crpix1 and crpix2 to determine the tile ID, and coordinates in degrees.
        naxis = hdr.get('NAXIS1')
        cdelt = abs(hdr.get('CDELT1'))
        hpx_ref_hdr = self._reference_header(naxis=naxis, cdelt=cdelt)
        hpx_ref_wcs = WCS(hpx_ref_hdr)

        crpix1 = hdr.get('CRPIX1')
        crpix2 = hdr.get('CRPIX2')
        crval1, crval2 = hpx_ref_wcs.wcs_pix2world(-crpix1, -crpix2 , 0)
        tileID = hp.lonlat_to_healpix(crval1 * units.deg, crval2 * units.deg, return_offsets=False)
        tileID = tileID - 1 #shifts by 1.

        # extract the RA and DEC for a specific pixel
        center = hp.healpix_to_lonlat(tileID) * units.deg
        RA, DEC = center.value

        self._logger.info(f'Derived RA is {RA} degrees and DEC is {DEC} degrees')
        c = SkyCoord(ra=RA * units.degree, dec=DEC * units.degree, frame='icrs')
        # coordinate string as "hhmm-ddmm"
        hmdm = self._RADEC_hms_dms_to_string(c)

        if mfs:
            outname = (
                f'{self._config.lookup.get('rename_prefix')}_{cenfreq}_{bmaj}_{hmdm}_{tileID}_t0_{stokesid}_'
                f'{self._config.lookup.get('rename_version')}.fits'
            )
        else:
            outname = (
                f'{self._config.lookup.get('rename_prefix')}_{cenfreq}_{bmaj}_{hmdm}_{tileID}_{stokesid}_'
                f'{self._config.lookup.get('rename_version')}.fits'
            )
        return outname

    def _reference_header(self, naxis, cdelt):
        """
        This is important as it allows us to properly determine correct pixel central pixel anywhere within the grid.

        NB: We use this header to convert the crpix1/2 in the header to tile ID, then degrees.

        :param cdelt the pixel size of the image in the grid. Must be the same as the one used for tiling.
        :param naxis number of pixels within each axis.
        """
        d = {
            'SIMPLE': 'T',
            'BITPIX': -32,
            'NAXIS': 2,
            'NAXIS1': naxis,
            'NAXIS2': naxis,
            'EXTEND': 'F',
            'CRPIX1': (naxis/2.0),
            'CRPIX2': (naxis/2.0) + 0.5,
            'PC1_1': 0.70710677,
            'PC1_2': 0.70710677,
            'PC2_1': -0.70710677,
            'PC2_2': 0.70710677,
            'CDELT1': -1 * cdelt,
            'CDELT2': cdelt,
            'CTYPE1': 'RA---HPX',
            'CTYPE2': 'DEC--HPX',
            'CRVAL1': 0.,
            'CRVAL2': 0.,
            'PV2_1': 4,
            'PV2_2': 3,
        }
        return fits.Header(d)

    def _prepare(self):
        """The files from Pawsey need to be renamed. Some of the metadata to rename the files is most easily found
        in the plane-level metadata that is calculated server-side.

        The sandbox POSSUM configuration calculates the plane-level metadata, but the production POSSUM configuration
        does not.

        The BINTABLE files require do not contain enough metadata to easily calculate plane-level metadata, so for
        those files, that must be calculated by this application. The position bounding box will be the HEALpix
        coordinates for n=32.
        """
        self._logger.debug('Begin _prepare')
        work = glob.glob('**/*.fits', root_dir=self._working_directory, recursive=True)
        for file_name in work:
            self._logger.debug(f'Working on {file_name}')
            found_storage_name = None
            for destination_uri in self._remote_metadata_reader.file_info.keys():
                if os.path.basename(destination_uri) == os.path.basename(file_name):
                    found_storage_name = PossumName(destination_uri)
                    self._logger.error(id(found_storage_name))
                    break

            original_fqn = os.path.join(self._working_directory, file_name)
            self._remote_metadata_reader.set_headers(found_storage_name, original_fqn)
            # TODO - not quite sure which header index to return :)
            headers = self._remote_metadata_reader.headers.get(found_storage_name.file_uri)
            if headers:
                renamed_file = self._find_new_file_name(headers[0], ('mfs' in found_storage_name.file_name))
                renamed_fqn = original_fqn.replace(os.path.basename(original_fqn), renamed_file)
                os.rename(original_fqn, renamed_fqn)
                self._logger.info(f'Renamed {original_fqn} to {renamed_fqn}.')
            else:
                self._logger.warning(f'Could not find headers for {file_name}')
        self._logger.debug('End _prepare')


def remote_execution():
    """When running remotely, do a time-boxed 2-stage execution:
    1. stage 1 - the work is to use rclone to retrieve files to a staging area
    2. stage 2 - with the files in the staging area, use the pipeline as usual to store the files and create and
                 store the CAOM2 records, thumbnails, and previews

    Stage 1 is controlled with the ExecutionUnitStateRunner.
    Stage 2 is controlled with a TodoRunner, that is created for every ExecutionUnitStateRunner time-box that brings
    over files.
    """
    logging.debug('Begin remote_execution')
    config = Config()
    config.get_executors()
    set_logging(config)
    observable = Observable(config)
    reporter = ExecutionReporter(config, observable)
    reporter.set_log_location(config)
    builder = EntryBuilder(PossumName)
    metadata_reader = RemoteRcloneMetadataReader(builder)
    organizer = ExecutionUnitOrganizeExecutes()
    clients = RCloneClients(config)
    StorageName.collection = config.collection
    StorageName.preview_scheme = config.preview_scheme
    StorageName.scheme = config.scheme
    kwargs = {
        'clients': clients,
        'reporter': reporter,
    }
    data_sources = []
    for entry in config.data_sources:
        data_source = RemoteIncrementalDataSource(
            config,
            entry,  # should look like "acacia_possum:pawsey0980" acacia_possum => rclone named config, pawsey0980 => root bucket
            metadata_reader,
            **kwargs,
        )
        data_source.reporter = reporter
        data_sources.append(data_source)
    runner = ExecutionUnitStateRunner(
        config,
        organizer,
        data_sources=data_sources,
        reporter=reporter,
    )
    result = runner.run()
    result |= runner.run_retry()
    runner.report()
    logging.debug('End remote_execution')
    return result
