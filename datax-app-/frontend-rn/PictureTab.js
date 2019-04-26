import React from 'react';
import { Button, StyleSheet, Image, View } from 'react-native';
import { ImagePicker, Permissions} from 'expo';
import { connect } from 'react-redux';

import { ENDPOINT_BASE } from './constants.js';
import Playlist from './Playlist.js';

class PictureTab extends React.Component {
  state = {
    image: null,
    playlist: [],
  };

  render() {
    let { image } = this.state;

    Permissions.askAsync(Permissions.CAMERA);
    Permissions.askAsync(Permissions.CAMERA_ROLL);

    return (
      <View style={{ flex: 1, alignItems: 'center', justifyContent: 'center' }}>
        <Button
          title="Pick an image from camera roll"
          onPress={this._pickImage}
        />
        {image &&
          <Image source={{ uri: image }} style={{ width: 200, height: 200 }} />}
        <Button
            title="Rock Spotify"
            onPress={() => this.rockSpotify()}
        />
        <Playlist styles={styles.playlist} playlist={this.state.playlist} />
      </View>
    );
  }

//   async fetchPlaylist() {
//     this.setState({ playlist: [] }); // Empty the playlist first

//     const response = await fetch(`${ENDPOINT_BASE}/playlist/create/text?text=${this.state.text}`, {
//         headers: {
//             'Authorization': this.props.userId,
//         },
//     });
//     const playlist = await response.json();
//     this.setState({ playlist });
// }

  _pickImage = async () => {
    let result = await ImagePicker.launchImageLibraryAsync({
      allowsEditing: true,
      aspect: [4, 3],
    });

    console.log(result);

    if (!result.cancelled) {
      this.setState({ image: result.uri });

      this.setState({ playlist: await uploadImageAsync(result.uri, this.props.userId) });
    }
  };
}


async function uploadImageAsync(uri, userId) {
    let apiUrl = `${ENDPOINT_BASE}/playlist/create/picture`;

    // Note:
    // Uncomment this if you want to experiment with local server
    //
    // if (Constants.isDevice) {
    //   apiUrl = `https://your-ngrok-subdomain.ngrok.io/upload`;
    // } else {
    //   apiUrl = `http://localhost:3000/upload`
    // }

    let uriParts = uri.split('.');
    let fileType = uriParts[uriParts.length - 1];

    let formData = new FormData();
    formData.append('photo', {
      uri,
      name: `photo.${fileType}`,
      type: `image/${fileType}`,
    });

    let options = {
      method: 'POST',
      body: formData,
      headers: {
        'Accept': 'application/json',
        'Content-Type': 'multipart/form-data',
        'Authorization': userId,
      },
    };

    console.log({ apiUrl, options });

    return (await fetch(apiUrl, options)).json();
  }

  const styles = StyleSheet.create({
    container: {
        flex: 1,
    },
    text: {
        height: 96,
        padding: 16,
        borderBottomWidth: 1,
        borderBottomColor: 'gray',
    },
    button: {
        borderBottomWidth: 1,
        borderBottomColor: 'gray',
    },
    playlist: {
        flex: 1,
    },
});

const mapStateToProps = (state) => {
    return {
        userId: state.userId
    };
};

const mapDispatchToProps = (dispatch) => {
    return {};
};

export default connect(mapStateToProps, mapDispatchToProps)(PictureTab);