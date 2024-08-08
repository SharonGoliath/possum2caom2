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

from mock import Mock, patch, PropertyMock

from possum2caom2.storage_name import PossumName
from possum2caom2 import fits2caom2_augmentation
from caom2.diff import get_differences
from caom2pipe import astro_composable as ac
from caom2pipe import manage_composable as mc
from caom2pipe import reader_composable as rdc

import glob
import logging
import os
import helpers


@patch('possum2caom2.possum_execute.RCloneClients')
@patch('caom2utils.data_util.get_local_headers_from_fits')
def test_main_app(header_mock, clients_mock, test_config, test_data_dir):
    # logging.getLogger().setLevel(logging.DEBUG)
    test_dir = f'{test_data_dir}/multi'
    test_expected_fqn = f'{test_dir}/PSM_944MHz_20asec_1034-5552_v1.expected.xml'
    actual_fqn = test_expected_fqn.replace('expected', 'actual')
    if os.path.exists(actual_fqn):
        os.unlink(actual_fqn)

    header_mock.side_effect = ac.make_headers_from_file
    clients_mock.metadata_client.read.return_value = None
    f_list = glob.glob(f'{test_dir}/*.header')
    observation = None
    test_observable = Mock()
    meta_producer_mock = PropertyMock(return_value='test_possum/0.0.0')
    type(test_observable).meta_producer = meta_producer_mock
    test_config.change_working_directory('/tmp')
    test_reporter = mc.ExecutionReporter(test_config, test_observable)
    for f_name in f_list:
        def _sandbox_mock(_, obs_id):
            sc2_name = f_name.replace('.fits.header', '.sc2.xml')
            if os.path.exists(sc2_name):
                return mc.read_obs_from_file(sc2_name)
            else:
                return None
        clients_mock.server_side_ctor_client.read.side_effect = _sandbox_mock

        storage_name = PossumName(entry=f_name)
        metadata_reader = rdc.FileMetadataReader()
        metadata_reader.set(storage_name)
        file_type = 'application/fits'
        metadata_reader.file_info[storage_name.file_uri].file_type = file_type
        kwargs = {
            'storage_name': storage_name,
            'metadata_reader': metadata_reader,
            'config': test_config,
            'clients': clients_mock,
            'reporter': test_reporter,
        }
        observation = fits2caom2_augmentation.visit(observation, **kwargs)

    if observation is None:
        assert False, f'Did not create observation for {test_expected_fqn}'
    else:
        if os.path.exists(test_expected_fqn):
            expected = mc.read_obs_from_file(test_expected_fqn)
            helpers.set_release_date_values(observation)
            compare_result = get_differences(observation, expected)
            if compare_result is not None:
                mc.write_obs_to_file(observation, actual_fqn)
                compare_text = '\n'.join([r for r in compare_result])
                msg = f'Differences found in observation {expected.observation_id}\n{compare_text}'
                raise AssertionError(msg)
        else:
            mc.write_obs_to_file(observation, actual_fqn)
            assert False, f'{test_expected_fqn} does not exist. Nothing to compare to for {test_expected_fqn}'
    # assert False  # cause I want to see logging messages
